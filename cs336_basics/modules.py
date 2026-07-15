# Linear Module for Transformer
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum, reduce
# from jaxtyping import Float, Int, Bool, Tuple, List, Dict, Optional
import math

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



class RoPEModule(nn.Module):
    """
    Rotary Positional Embedding (RoPE) module.
    Applies rotary positional embeddings to the input tensor.
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device |None = None, dtype: torch.dtype |None = None):
        # Initialize the RoPE module with the given parameters.
        # theta: The base frequency for the rotary embeddings.
        # d_k: The dimension of the key/query vectors (must be even).
        # max_seq_len: The maximum sequence length for which to precompute the embeddings.
        
        super().__init__()
        if theta <= 0:
            raise ValueError("theta must be positive for RoPE.")
        self.theta = theta
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute the inverse frequencies for the rotary embeddings.
        pair_idx = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = self.theta ** (- pair_idx / d_k) # ( d_k // 2, )
        seq_idx = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        
        # angles: (max_seq_len, d_k // 2)
        angles = torch.einsum("i,j->i j", seq_idx, inv_freq)
        
        cos_emb = torch.cos(angles)  # (max_seq_len, d_k // 2)
        sin_emb = torch.sin(angles)  # (max_seq_len, d_k // 2)

        # Store the cosine and sine embeddings as buffers for later use.
        self.register_buffer("cos_emb", cos_emb, persistent=False)
        self.register_buffer("sin_emb", sin_emb, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len), specifying the position of x
        # output: (..., seq_len, d_k)
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Last dimension of x must be {self.d_k}, but got {x.shape[-1]}.")
        if token_positions.min() < 0 or token_positions.max() >= self.max_seq_len:
            raise ValueError(f"token_positions must be in the range [0, {self.max_seq_len}).")
        pos = token_positions.to(device=x.device, dtype=torch.long)
        cos = self.cos_emb.index_select(0, pos.reshape(-1)).reshape(*pos.shape, -1)  # (..., seq_len, d_k // 2)
        sin = self.sin_emb.index_select(0, pos.reshape(-1)).reshape(*pos.shape, -1)  # (..., seq_len, d_k // 2)

        x_fp32 = x.to(torch.float32)
        x_even = x_fp32[..., ::2]  # (..., seq_len, d_k // 2)
        x_odd = x_fp32[..., 1::2]  # (..., seq_len, d_k // 2)

        # make cos and sin broadcastable to x_even and x_odd for inputs like (b, h, s, d_k)
        while cos.dim() < x_even.dim():
            cos = cos.unsqueeze(cos.dim() -2)
            sin = sin.unsqueeze(sin.dim() -2)
        
        # Apply the rotary transformation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave the even and odd parts back together
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)  # (..., seq_len, d_k)
        return x_rotated.to(x.dtype)
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the softmax of the input tensor along the specified dimension.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_sum



def dot_product_attention(q: torch.Tensor, 
                          k: torch.Tensor, 
                          v: torch.Tensor, 
                          mask: torch.Tensor | None = None
                          ) -> torch.Tensor:
    """
    Computes the dot product attention.
    q, k: (batch_size, ... , seq_len, d_k)
    v: (batch_size, ... , seq_len, d_v)
    mask: (seq_len, seq_len), optional, 
        boolean mask where True indicates positions should attend to.
    Returns:
        The attention output tensor of shape (batch_size, ... , seq_len, d_v).
    """
    # Compute the dot product between queries and keys
    attention_scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(q.shape[-1])  # (..., seq_len, seq_len)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to get the attention weights
    attention_weights = softmax(attention_scores, dim=-1)

    # Compute the weighted sum of values
    output = torch.matmul(attention_weights, v)  # (..., seq_len, d_v)

    return output


class MultiheadSelfAttentionModule(nn.Module):
    """
    Multihead Self-Attention module.
    """
    def __init__(self, d_model:int, num_heads:int, max_seq_len: int|None=None, theta: float|None=None,  device: torch.device |None = None, dtype: torch.dtype |None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len 
        self.theta = theta

        # Define the linear layers for query, key, value, and output projections
        self.qkv_proj = LinearModule(in_features=d_model, out_features=3*d_model, device=device, dtype=dtype)
        self.o_proj = LinearModule(in_features=d_model, out_features=d_model, device=device, dtype=dtype)

        # Define the RoPE module for positional embeddings
        if self.max_seq_len is not None and self.theta is not None:
            self.rope = RoPEModule(theta=self.theta, d_k=self.d_k, max_seq_len=self.max_seq_len, device=device, dtype=dtype)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        # token_positions: (batch_size, seq_len), optional
        # output: (batch_size, seq_len, d_model)

        # Project the input to queries, keys, and values
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3*d_model)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multihead attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)  # (batch_size, num_heads, seq_len, d_k)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)  # (batch_size, num_heads, seq_len, d_k)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)  # (batch_size, num_heads, seq_len, d_k)

        # Apply RoPE if token_positions are provided
        if token_positions is not None and self.rope is not None:
            q = self.rope(q, token_positions)  # (batch_size, num_heads, seq_len, d_k)
            k = self.rope(k, token_positions)  # (batch_size, num_heads, seq_len, d_k)
        
        # causal mask for self-attention
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)  # (seq_len, seq_len)

        # Compute the attention output
        attn_output = dot_product_attention(q, k, v, mask=~causal_mask)  # (batch_size, num_heads, seq_len, d_k)
        # Reshape the attention output back to the original shape
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")  # (batch_size, seq_len, d_model)

        # Project the attention output back to the original dimension
        output = self.o_proj(attn_output)  # (batch_size, seq_len, d_model)
        return output