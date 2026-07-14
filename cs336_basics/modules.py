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
        # x: (..., in_features)
        # weight: (out_features, in_features)
        # output: (..., out_features)
        x = einsum(x, self.weight, "... i, o i -> ... o")
        return x
    


class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device: torch.device |None = None, dtype: torch.dtype |None = None):
        super().__init__()
        std = 1.0
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, std=std, a=-3*std, b=3*std)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (...,)
        # self.embedding: (num_embeddings, embedding_dim)
        # output: (batch, seq_len, embedding_dim)
        x = self.embedding[token_ids]
        return x


class RMSNormModule(nn.Module):
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
    

class SwiGLUModule(nn.Module):
    """
    SwiGLU activation function module. Combines the SiLU activation with GLU gating mechanism.
    """
    def __init__(self, d_model:int, d_ff:int | None = None, device: torch.device |None = None, dtype: torch.dtype |None = None):
        super().__init__()
        # W1, W2, W3 are the weights for the linear transformations
        # W1, W3 (d_ff, d_model), W2 (d_model, d_ff)
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._default_d_ff(d_model)
        self.W1 = LinearModule(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.W2 = LinearModule(in_features=self.d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = LinearModule(in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype)


    def _default_d_ff(self, d_model: int, round_base: int =64) -> int:
        """
        Computes the default feedforward dimension based on the model dimension.
        The default is 8/3* d_model, rounded up to the nearest multiple of 64
        """
        return round_base * ((8 * d_model + 3 * round_base - 1) // (3 * round_base))
    
    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SiLU activation function.
        """
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (d_model)
        # apply silu activation
        x1 = self.W1(x)  # (d_ff)
        x2 = self.W3(x)  # (d_ff)
        x1 = self._silu(x1)  # (d_ff)
        x = x1 * x2  # (d_ff)
        x = self.W2(x)  # (d_model)
        return x

