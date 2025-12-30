# models/qor/LSOformer/lsoformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import GraphEncoder
from .recipe_encoder import RecipeEncoder
from .transformer_decoder import TransformerDecoderModule, subsequent_mask

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        # x: [B, d] or [B, M, d]
        if x.dim() == 3:
            B, M, D = x.shape
            return self.net(x.view(B*M, D)).view(B, M)
        else:
            return self.net(x).squeeze(-1)

class LSOformer(nn.Module):
    def __init__(self,
                 node_emb_dim=16,
                 gcn_hidden=32,
                 num_gcn_layers=2,
                 max_levels=200,
                 recipe_vocab_size=10,
                 recipe_d_model=64,
                 recipe_len=20,
                 decoder_layers=4,
                 decoder_heads=4,
                 decoder_ff=256,
                 dropout=0.1,
                 alpha_traj=1.0):
        super().__init__()
        self.graph_enc = GraphEncoder(node_emb_dim, gcn_hidden, num_gcn_layers, max_levels)
        self.recipe_enc = RecipeEncoder(recipe_vocab_size, d_model=recipe_d_model, recipe_len=recipe_len)
        self.decoder = TransformerDecoderModule(d_model=recipe_d_model, nhead=decoder_heads,
                                                num_layers=decoder_layers, dim_feedforward=decoder_ff, dropout=dropout)
        self.recipe_len = recipe_len
        self.alpha_traj = alpha_traj
        # MLP to map decoder hidden to QoR per step (decoder produces per-step embedding dim = recipe_d_model)
        self.mlp = MLPRegressor(recipe_d_model, hidden=recipe_d_model*2)

    def forward(self, batch_data):
        """
        batch_data: torch_geometric Batch with fields:
           - edge_index
           - batch
           - node_type (N,)
           - num_inverted_predecessors (N,)
           - synVec (B, M) LongTensor
        returns:
           final_pred: [B] final QoR preds
           traj_pred: [B, M] per-step QoR preds
        """
        device = batch_data.synVec.device
        # Graph sequence H: [B, S, 2*gcn_hidden]
        H = self.graph_enc(batch_data).to(device)
        B, S, Gdim2 = H.shape
        # project graph memory to decoder d_model if needed
        if Gdim2 != self.recipe_enc.embedding.embedding_dim:
            # linear to match dimensions
            proj = getattr(self, "_memory_proj", None)
            if proj is None:
                self._memory_proj = nn.Linear(Gdim2, self.recipe_enc.embedding.embedding_dim).to(device)
                proj = self._memory_proj
            memory = proj(H)  # [B, S, d_model]
        else:
            memory = H
        # Permute for transformer: seq-first [S, B, d_model]
        memory = memory.permute(1, 0, 2)

        # recipe encoding
        tgt = self.recipe_enc(batch_data.synVec.to(device))  # [B, M, d_model]
        # tgt to seq-first [M, B, d_model]
        tgt = tgt.permute(1, 0, 2)
        # build causal mask for target
        Mlen = tgt.size(0)
        mask = subsequent_mask(Mlen).to(device)  # upper triangular True where should be masked
        # transformer expects float mask with -inf where masked or None. PyTorch accepts bool mask for tgt_mask where True=masked.
        out = self.decoder(tgt, memory, tgt_mask=mask)
        # out: [M, B, d_model] -> permute to [B, M, d_model]
        out = out.permute(1, 0, 2)
        # predict trajectory with MLP
        traj = self.mlp(out)  # [B, M]
        final = traj[:, -1]  # last token as final QoR
        return final, traj
