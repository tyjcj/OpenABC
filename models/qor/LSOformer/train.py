import os
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
from models.qor.LSOformer.lsoformer import LSOformer
from models.qor.LSOformer.netlistDataset import LSOListDataset

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import random_split
import os.path as osp
import pickle
import sys
import matplotlib.pyplot as plt

datasetDict = {
    'set1': ["train_data_set1.csv", "test_data_set1.csv"],
    'set2': ["train_data_set2.csv", "test_data_set2.csv"],
    'set3': ["train_data_mixmatch_v1.csv", "test_data_mixmatch_v1.csv"]
}

DUMP_DIR = None
criterion = torch.nn.MSELoss()


def plotChart(x, y, xlabel, ylabel, leg_label, title):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=leg_label)
    plt.legend(loc='best', ncol=2, shadow=True, fancybox=True).get_frame().set_alpha(0.5)
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.title(title, weight='bold')
    plt.savefig(osp.join(DUMP_DIR, title + '.png'), fmt='png', bbox_inches='tight')


def train(model, device, dataloader, optimizer, alpha=1.0):
    epochLoss = AverageMeter()
    model.train()
    for _, batch in enumerate(tqdm(dataloader, desc="Iteration", file=sys.stdout)):
        batch = batch.to(device)
        optimizer.zero_grad()
        final_pred, traj_pred = model(batch)
        loss_final = criterion(final_pred, batch.target_final.view(-1).float())
        loss_traj = criterion(traj_pred, batch.target_traj.float())
        loss = loss_final + alpha * loss_traj
        loss.backward()
        optimizer.step()
        epochLoss.update(loss.item(), batch.num_graphs)
    return epochLoss.avg


def evaluate(model, device, dataloader):
    model.eval()
    validLoss = AverageMeter()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader, desc="Iteration", file=sys.stdout)):
            batch = batch.to(device)
            final_pred, traj_pred = model(batch)
            mseVal = mse(final_pred, batch.target_final.view(-1).float())
            validLoss.update(mseVal, batch.num_graphs)
    return validLoss.avg


def main():
    parser = argparse.ArgumentParser(description='LSOformer training on OpenABC-D')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lp', type=int, default=1, help='Learning problem ID (lp folder name)')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--dataset', type=str, default="set1", help='set1/set2/set3')
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--target', type=str, default="delay", help='QoR target (delay/area/nodes)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for trajectory loss')
    args = parser.parse_args()

    datasetChoice = args.dataset
    batchSize = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    learningProblem = args.lp
    targetLbl = args.target
    alpha = args.alpha

    global DUMP_DIR
    DUMP_DIR = args.rundir
    os.makedirs(DUMP_DIR, exist_ok=True)

    ROOT_DIR = args.datadir
    trainDS = LSOListDataset(root=osp.join(ROOT_DIR, "lp" + str(learningProblem)),
                             filePath=datasetDict[datasetChoice][0])
    testDS = LSOListDataset(root=osp.join(ROOT_DIR, "lp" + str(learningProblem)),
                            filePath=datasetDict[datasetChoice][1])

    # load target statistics
    with open(osp.join(ROOT_DIR, 'synthesisStatistics.pickle'), 'rb') as f:
        targetStats = pickle.load(f)
    meanVarTargetDict = computeMeanAndVarianceOfTargets(targetStats, targetVar=targetLbl)
    trainDS.transform = transforms.Compose([lambda data: addNormalizedTargets(data, targetStats, meanVarTargetDict, targetVar=targetLbl)])
    testDS.transform = transforms.Compose([lambda data: addNormalizedTargets(data, targetStats, meanVarTargetDict, targetVar=targetLbl)])

    # Model
    model = LSOformer(recipe_vocab_size=8, recipe_len=20, recipe_d_model=64, decoder_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    device = getDevice()
    model = model.to(device)

    # Split train/valid
    train_DS, valid_DS = random_split(trainDS, [int(0.8 * len(trainDS)), len(trainDS) - int(0.8 * len(trainDS))])
    train_dl = DataLoader(train_DS, shuffle=True, batch_size=batchSize, num_workers=4)
    valid_dl = DataLoader(valid_DS, shuffle=True, batch_size=batchSize, num_workers=4)
    test_dl = DataLoader(testDS, shuffle=True, batch_size=batchSize, num_workers=4)

    # Training loop
    bestVal, bestEpoch = float('inf'), 0
    valid_curve, train_curve = [], []
    for ep in range(1, num_epochs + 1):
        print(f"\nEpoch {ep}/{num_epochs}")
        trainLoss = train(model, device, train_dl, optimizer, alpha)
        validLoss = evaluate(model, device, valid_dl)
        print({'Train': trainLoss, 'Valid': validLoss})

        if validLoss < bestVal:
            bestVal, bestEpoch = validLoss, ep
            torch.save(model.state_dict(), osp.join(DUMP_DIR, f'lsoformer-epoch{ep}-val{validLoss:.3f}.pt'))

        valid_curve.append(validLoss)
        train_curve.append(trainLoss)
        scheduler.step(validLoss)

    # reload best
    model.load_state_dict(torch.load(osp.join(DUMP_DIR, f'lsoformer-epoch{bestEpoch}-val{bestVal:.3f}.pt')))

    # save curves
    with open(osp.join(DUMP_DIR, 'valid_curve.pkl'), 'wb') as f:
        pickle.dump(valid_curve, f)
    with open(osp.join(DUMP_DIR, 'train_loss.pkl'), 'wb') as f:
        pickle.dump(train_curve, f)

    # final evaluation
    testLoss = evaluate(model, device, test_dl)
    print("********************")
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Best Epoch: {bestEpoch}, Valid Loss: {bestVal:.4f}, Test Loss: {testLoss:.4f}")
    print("********************")


if __name__ == "__main__":
    main()
