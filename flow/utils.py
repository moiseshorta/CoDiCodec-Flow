"""Misc utilities: device detection, attention masks, logging helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------

def best_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available torch device.

    Order: explicit preference -> CUDA -> Apple MPS -> CPU.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device, dtype: torch.dtype = torch.bfloat16, enabled: bool = True):
    """Return an autocast context that works on cuda/mps/cpu.

    On MPS, bf16 has uneven op coverage; the caller can pass enabled=False to
    fall back to fp32 if a particular op blows up.
    """
    if not enabled:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=False)
    if device.type in ("cuda", "cpu", "mps"):
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    return torch.autocast(device_type="cpu", dtype=dtype, enabled=False)


# ---------------------------------------------------------------------------
# Attention masks
# ---------------------------------------------------------------------------

def block_causal_mask(seq_len: int, block_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Build an additive attention mask for block-causal self-attention.

    Tokens are grouped into consecutive blocks of `block_size`. Within a block
    attention is bidirectional; across blocks it is causal (token i can attend
    to tokens in blocks <= block(i)).

    Returns a `[seq_len, seq_len]` float mask where 0 = allowed and -inf = masked.
    Suitable for `torch.nn.functional.scaled_dot_product_attention(attn_mask=...)`.
    """
    idx = torch.arange(seq_len, device=device)
    block_idx = idx // block_size  # [L]
    allowed = block_idx.unsqueeze(0) <= block_idx.unsqueeze(1)  # [L, L]
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


# ---------------------------------------------------------------------------
# Reproducibility / logging
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "flow", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def human_int(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.2f}K"
    return str(n)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
