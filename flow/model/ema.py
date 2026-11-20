"""Exponential Moving Average of model parameters.

Standard implementation: a parallel copy of the parameters, kept on the
model's device, updated after every optimizer step as

    ema_p = d * ema_p + (1 - d) * p,    d = min(decay, (1 + n) / (10 + n))

The warmup on `d` keeps early checkpoints / previews from being dominated by
the random init. Buffers are copied verbatim.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.num_updates = 0
        self.device = next(model.parameters()).device
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone().float()
            for name, p in model.named_parameters() if p.requires_grad
        }
        self.buffers: dict[str, torch.Tensor] = {
            name: b.detach().clone() for name, b in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        params = [p.detach() for name, p in model.named_parameters() if name in self.shadow]
        torch._foreach_lerp_(list(self.shadow.values()), params, 1.0 - d)
        for name, b in model.named_buffers():
            self.buffers[name] = b.detach().clone()

    def state_dict(self) -> dict:
        sd = {}
        sd.update({f"params.{k}": v.cpu() for k, v in self.shadow.items()})
        sd.update({f"buffers.{k}": v.cpu() for k, v in self.buffers.items()})
        sd["__decay__"] = torch.tensor(self.decay)
        sd["__num_updates__"] = torch.tensor(self.num_updates)
        return sd

    def load_state_dict(self, sd: dict) -> None:
        self.shadow = {}
        self.buffers = {}
        # Checkpoints from before the warmup existed: treat the warmup as long over.
        self.num_updates = 10**9
        for k, v in sd.items():
            if k == "__decay__":
                self.decay = float(v.item())
            elif k == "__num_updates__":
                self.num_updates = int(v.item())
            elif k.startswith("params."):
                self.shadow[k[len("params."):]] = v.to(self.device, torch.float32)
            elif k.startswith("buffers."):
                self.buffers[k[len("buffers."):]] = v.to(self.device)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite model parameters with EMA values (in-place)."""
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name])
        for name, b in model.named_buffers():
            if name in self.buffers:
                b.data.copy_(self.buffers[name])
