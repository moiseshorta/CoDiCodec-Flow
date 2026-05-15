"""Exponential Moving Average of model parameters.

Standard implementation: a parallel state dict updated as
    ema_p = decay * ema_p + (1 - decay) * p
after every optimizer step. Buffers are copied verbatim.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Keep a CPU shadow to save MPS/CUDA memory.
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().float().cpu()
        self.buffers: dict[str, torch.Tensor] = {}
        for name, b in model.named_buffers():
            self.buffers[name] = b.detach().clone().cpu()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow[name]
            new_val = p.detach().float().cpu()
            shadow.mul_(d).add_(new_val, alpha=1.0 - d)
        for name, b in model.named_buffers():
            self.buffers[name] = b.detach().clone().cpu()

    def state_dict(self) -> dict:
        sd = {}
        sd.update({f"params.{k}": v for k, v in self.shadow.items()})
        sd.update({f"buffers.{k}": v for k, v in self.buffers.items()})
        sd["__decay__"] = torch.tensor(self.decay)
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {}
        self.buffers = {}
        for k, v in sd.items():
            if k == "__decay__":
                self.decay = float(v.item())
            elif k.startswith("params."):
                self.shadow[k[len("params.") :]] = v
            elif k.startswith("buffers."):
                self.buffers[k[len("buffers.") :]] = v

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite model parameters with EMA values (in-place)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(dtype=p.dtype, device=p.device))
        for name, b in model.named_buffers():
            if name in self.buffers:
                b.data.copy_(self.buffers[name].to(dtype=b.dtype, device=b.device))
