import torch
import torch.nn as nn
import numpy as np
import os, typing
import numpy.typing as npt
from einops import rearrange, einsum, reduce
from typing import Optional
from collections.abc import Callable, Iterable
import math
import cs336_basics.modules as modules
from cs336_basics.config import get_default_config
import time

def load_dataset_with_memmap(path: str, dtype: np.dtype) -> npt.NDArray:
    """
    Loads a dataset from a given path. The dataset is expected to be a binary file containing 32-bit integers. 
    The function uses memory mapping to efficiently load the dataset without reading the entire file into memory at once.

    Args:
        path (str): Path to the dataset file.
        dtype (np.dtype): The data type of the elements in the dataset.
    """

    return np.memmap(path, dtype=dtype, mode='r')


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    """
    Converts a string representation of a data type to a corresponding PyTorch data type.

    Args:
        dtype_str (str): String representation of the data type. 
                         Supported values: 'float32', 'float16', 'bfloat16'.

    Returns:
        torch.dtype: Corresponding PyTorch data type.
    """
    if dtype_str in ["float32", "fp32"]:
        return torch.float32
    elif dtype_str in ["float16", "fp16"]:
        return torch.float16
    elif dtype_str in ["bfloat16", "bf16"]:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype string: {dtype_str}. Supported values are 'float32', 'float16', 'bfloat16'.")

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Sets the learning rate for all parameter groups in the given optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate needs to be set.
        lr (float): The new learning rate to be set.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

@torch.no_grad()
def estimate_loss(model: nn.Module, dataset: npt.NDArray, cfg) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.training.eval_batches):
        x, y = modules.get_batch(x=dataset, batch_size=cfg.training.batch_size, context_length=cfg.model.context_length, device=cfg.model.device)
        logits = model(x)
        loss = modules.cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    cfg = get_default_config()
    
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    wandb = None
    if cfg.wandb.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb.project_name, name=cfg.wandb.run_name)
    
    os.makedirs(os.path.dirname(cfg.training.ckpt_path) or ".", exist_ok=True)

    train_mm = load_dataset_with_memmap(cfg.dataset.train_path, dtype=cfg.dataset.np_dtype)
    val_mm = load_dataset_with_memmap(cfg.dataset.val_path, dtype=cfg.dataset.np_dtype)

    device = torch.device(cfg.model.device)
    model_dtype = torch_dtype_from_str(cfg.torch_dtype)

    d_ff = cfg.model.d_ff if cfg.model.d_ff is not None else 4 * cfg.model.d_model

    model = modules.Transformer(
        vocab_size=cfg.dataset.vocab_size,
        d_model=cfg.model.d_model,
        d_ff=d_ff,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        context_length=cfg.model.context_length,
        rope_theta=cfg.model.rope_theta,
        max_seq_len=cfg.model.max_seq_len,
        rmsnorm_eps=cfg.model.rmsnorm_eps,
        dtype=model_dtype,
    ).to(device)

    optimizer = modules.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr_max,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    start_iter = 0
    if cfg.training.resume_from_ckpt is not None and os.path.exists(cfg.training.resume_from_ckpt):
        start_iter = modules.load_checkpoint(src=cfg.training.resume_from_ckpt, model=model, optimizer=optimizer)
    
    best_val_loss = float('inf')
    last_log_t = time.time()

    # training loop
    for iter in range(start_iter, cfg.training.num_iters):
        x, y = modules.get_batch(x=train_mm, batch_size=cfg.training.batch_size, context_length=cfg.model.context_length, device=device)
        logits = model(x)
        loss = modules.cross_entropy_loss(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if cfg.optimizer.grad_clip > 0:
            modules.gradient_clipping(model.parameters(), cfg.optimizer.grad_clip, cfg.optimizer.eps)
        
        optimizer.step()
        if (iter+1) % cfg.training.log_interval == 0:
            now = time.time()
            dt = max(now - last_log_t, 1e-8)
            tok_s = (cfg.training.batch_size * cfg.model.context_length * cfg.training.log_interval) / dt
            msg = f"iter {iter+1}: train loss {loss.item():.4f}, {tok_s:.2f} tokens/s"
            print(msg)
            last_log_t = now

        if (iter+1) % cfg.training.eval_interval == 0:
            val_loss = estimate_loss(model, val_mm, cfg)
            msg = f"iter {iter+1}: validation loss {val_loss:.4f}"
            print(msg)
            if wandb is not None:
                wandb.log({"train_loss": loss.item(), "val_loss": val_loss, "iteration": iter+1})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = cfg.training.ckpt_path.replace(".pt", f"_best.pt")
                modules.save_checkpoint(model=model, optimizer=optimizer, iteration=iter+1, out=best_ckpt_path)
                print(f"Saved checkpoint at iter {iter+1} with validation loss {val_loss:.4f}")
        
        if (iter+1) % cfg.training.ckpt_interval == 0:
            modules.save_checkpoint(model=model, optimizer=optimizer, iteration=iter+1, out=cfg.training.ckpt_path)
            print(f"Saved checkpoint at iter {iter+1}")

    modules.save_checkpoint(model=model, optimizer=optimizer, iteration=cfg.training.num_iters, out=cfg.training.ckpt_path)
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
