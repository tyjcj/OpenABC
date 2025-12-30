# models/qor/LSOformer/netlistDataset.py
import os.path as osp
import torch
from zipfile import ZipFile
import pandas as pd
from torch_geometric.data import Dataset

class LSOListDataset(Dataset):
    def __init__(self, root, filePath, transform=None, pre_transform=None):
        self.filePath = osp.join(root, filePath)
        super(LSOListDataset, self).__init__(root, transform, pre_transform)

    @property
    def processed_file_names(self):
        fileDF = pd.read_csv(self.filePath)
        return fileDF['fileName'].tolist()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # same zipped .pt file layout as original NetlistGraphDataset
        filePathArchive = osp.join(self.processed_dir, self.processed_file_names[idx])
        filePathName = osp.basename(osp.splitext(filePathArchive)[0])
        with ZipFile(filePathArchive) as myzip:
            with myzip.open(filePathName) as myfile:
                data = torch.load(myfile)

        # Expect data is a torch_geometric.data.Data object or dict with:
        # - node_type (N,), num_inverted_predecessors (N,), edge_index, batch
        # - synVec: recipe ints [B, M] OR single recipe [M] (if single graph per file)
        # - target_final: scalar or [B]
        # - target_traj: [B, M] optional
        # If not present, you may need to build synVec/targets from synthID2Vec.pickle / synthesisStatistics.pickle
        # Here we try to normalize shape:
        if hasattr(data, 'synVec'):
            syn = data.synVec
            if syn.dim() == 1:
                syn = syn.unsqueeze(0)  # [1, M]
            data.synVec = syn.long()
        else:
            # missing: create dummy zeros (will prevent training unless user fills)
            data.synVec = torch.zeros((1,20), dtype=torch.long)

        # ensure targets
        if not hasattr(data, 'target_final'):
            if hasattr(data, 'y'):
                data.target_final = data.y
            else:
                data.target_final = torch.tensor([0.0])

        # trajectory optional
        if not hasattr(data, 'target_traj'):
            # try to build a repeated sequence: fill with final
            data.target_traj = data.target_final.unsqueeze(-1).repeat(1,20) if data.target_final.dim()==2 else data.target_final.unsqueeze(0).repeat(1,20)

        return data
