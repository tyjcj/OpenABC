# models/qor/LSOformer/transformer_decoder.py
import torch
import torch.nn as nn

def subsequent_mask(sz):
    # causal mask: allow attending to current and past
    attn_shape = (sz, sz)
    subsequent = torch.triu(torch.ones(attn_shape), diagonal=1).bool()
    return subsequent  # True where mask (to be filled with -inf)

class TransformerDecoderModule(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           activation='relu')
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        """
        tgt: [M, B, d_model] (PyTorch transformer expects seq-first)
        memory: [S, B, d_model]  (S = number of graph levels)
        """
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)
        return out
