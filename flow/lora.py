"""LoRA adapters on top of a trained FlowDiT.

Train: `train --lora-base weights/ema.pt` freezes the base model (weights and
latent normalization) and wraps the attention and MLP projections of every
block in `LoRALinear`, so only the low-rank factors A, B learn. A run writes
`<out_dir>/<out_dir name>.pt` (latest checkpoint) and `<out_dir name>_best.pt` (best
validation step): the EMA of A and B, 7 MB at rank 16, 55 MB at rank 128.

Play: `LoraBank` loads any number of those files over one base model and
rebuilds the targeted weights in place, W = W_base + sum_i s_i * B_i @ A_i.
A switch costs a few small matmuls between chunks; generation runs at base speed.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_logger

logger = get_logger("flow.lora")

TARGETS = ("attn.qkv", "attn.out", "mlp.w1", "mlp.w2")


def rank_scale(rank: int) -> float:
    """rsLoRA scaling (Kalajdzievski 2023): one learning rate works at any rank; 1.0 at rank 16."""
    return 4.0 / math.sqrt(rank)


class LoRALinear(nn.Module):
    """Frozen `base` plus a trainable low-rank update scale * B @ A (B starts at zero)."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(rank, base.in_features, device=dev), a=math.sqrt(5)))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        self.scale = rank_scale(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale


def export(params: dict) -> dict:
    """File payload from {'<module>.A': A, '<module>.B': B}: the scale folded into B, so a
    LoRA file holds the plain update B @ A whatever its rank."""
    return {k: (v * rank_scale(params[k[:-1] + "A"].shape[0]) if k.endswith(".B") else v).cpu()
            for k, v in params.items()}


def add_lora(model: nn.Module, rank: int) -> None:
    """Freeze `model`, then wrap every targeted Linear in a LoRALinear."""
    model.requires_grad_(False)
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name.endswith(TARGETS):
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, LoRALinear(mod, rank))


class LoraBank:
    """LoRA files over one base model, blended live with `set({name: strength})`."""

    def __init__(self, model: nn.Module, paths):
        dev = next(model.parameters()).device
        self.model = model
        # ponytail: named by file stem; two files with the same stem, the last one wins.
        self.loras = {Path(p).stem: torch.load(p, map_location=dev, weights_only=True)["lora"] for p in paths}
        mods = {k.rpartition(".")[0] for lora in self.loras.values() for k in lora}
        self.base = {m: model.get_submodule(m).weight.detach().clone() for m in mods}
        self.mix: dict = {}
        self.strength = 1.0  # used by cycle() / nudge()

    @torch.no_grad()
    def set(self, mix: dict) -> None:
        """{lora name: strength} applied together; {} restores the base model."""
        unknown = set(mix) - set(self.loras)
        if unknown:
            raise KeyError(f"unknown LoRA {sorted(unknown)}; loaded: {sorted(self.loras)}")
        for m, w0 in self.base.items():
            w = self.model.get_submodule(m).weight
            w.copy_(w0)
            for name, s in mix.items():
                lora = self.loras[name]
                if m + ".A" in lora:
                    w.addmm_(lora[m + ".B"], lora[m + ".A"], alpha=float(s))
        self.mix = {n: float(s) for n, s in mix.items()}
        logger.info("LoRA: %s", ", ".join(f"{n} x{s:.2f}" for n, s in self.mix.items()) or "off (base model)")

    def cycle(self) -> None:
        """Round robin: each LoRA alone at `strength`, then the base model."""
        names = [None, *self.loras]
        cur = next(iter(self.mix)) if len(self.mix) == 1 else None
        nxt = names[(names.index(cur) + 1) % len(names)]
        self.set({nxt: self.strength} if nxt else {})

    def nudge(self, delta: float) -> None:
        self.strength = round(max(0.0, self.strength + delta), 2)
        self.set({n: self.strength for n in self.mix})
