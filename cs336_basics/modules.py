# Linear Module for Transformer
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum, reduce

class LinearModule(nn.Module):
    def __init__(self, in_features:int, out_features:int, device: torch.device |None = None, dtype: torch.dtype |None = None):
        super().__init__()
        std = np.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
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
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, std=std, a=-3*std, b=3*std)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        # self.embedding: (num_embeddings, embedding_dim)
        # output: (batch, seq_len, embedding_dim)
        x = self.embedding[token_ids]
        return x


class RMSLayerNorm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device: torch.device |None = None, dtype: torch.dtype |None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # output: (batch, seq_len, d_model)
        in_type = x.dtype
        x = x.to(torch.float32)
        mean_square = reduce(x ** 2, "b s d -> b s 1", "mean")
        x_norm = x / torch.sqrt(mean_square + self.eps)
        result = x_norm * self.weight
        return result.to(in_type)