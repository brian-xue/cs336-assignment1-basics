# Linear Module for Transformer
import torch
import torch.nn as nn
import torch.nn.Parameter as P
import numpy as np
from einops import rearrange, einsum, reduce

class LinearModule(nn.Module):
    def __init__(self, in_features:int, out_features:int, device: torch.device |None = None, dtype: torch.dtype |None = None):
        super().__init__()
        std = np.sqrt(2 / (in_features + out_features))
        self.weight = P(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_features)
        # weight: (out_features, in_features)
        # output: (batch, seq_len, out_features)
        x = einsum(x, self.weight, "b s i, o i -> b s o")
        return x
    


class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device: torch.device |None = None, dtype: torch.dtype |None = None):
        super().__init__()
        std = 1.0
        self.embedding = P(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, std=std, a=-3*std, b=3*std)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        # self.embedding: (num_embeddings, embedding_dim)
        # output: (batch, seq_len, embedding_dim)
        x = self.embedding[token_ids]
        return x
