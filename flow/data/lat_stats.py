"""Latent normalization statistics.

For Conditional Flow Matching to converge cleanly, the data should look like
N(0, I) at the boundary t=0. CoDiCodec's `atanh / sigma_rescale` post-processing
is *almost* there (mean ≈ 0.08, std ≈ 0.93 in our checks), but the residual
shift/scale is non-trivial enough that

    1. sampling from N(0, 1) at t=1 implicitly assumes the model has learned
       to translate/rescale the implicit data prior, which wastes capacity, and
    2. the small mean shift means the velocity field at low `t` predicts a
       non-zero average, which is hard to learn from random noise inputs.

We follow SD3 / Stable Audio 2: compute per-channel mean and std on the train
shards and normalize to unit-Gaussian before training. Sampling integrates in
normalized space and denormalizes the final output.

Stats are stored as a single `.pt` file in the data directory so they are
shared between training and inference; they're also baked into model
checkpoints as buffers so a downloaded checkpoint is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from ..config import CODEC_LATENT_DIM
from ..utils import get_logger

logger = get_logger("flow.data.lat_stats")


@dataclass
class LatStats:
    mean: torch.Tensor  # [latent_dim]
    std: torch.Tensor   # [latent_dim]
    count: int          # number of latent tokens used to compute the stats

    def to_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "count": int(self.count)}

    @classmethod
    def from_dict(cls, obj: dict) -> "LatStats":
        return cls(
            mean=torch.as_tensor(obj["mean"], dtype=torch.float32),
            std=torch.as_tensor(obj["std"], dtype=torch.float32),
            count=int(obj["count"]),
        )

    @classmethod
    def identity(cls, dim: int = CODEC_LATENT_DIM) -> "LatStats":
        return cls(
            mean=torch.zeros(dim, dtype=torch.float32),
            std=torch.ones(dim, dtype=torch.float32),
            count=0,
        )


def _list_shards(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*.pt"):
        if p.name == "lat_stats.pt":
            continue
        out.append(p)
    out.sort()
    return out


def compute_lat_stats(data_dir: str | Path, shard_paths: List[Path] | None = None) -> LatStats:
    """Compute per-channel mean/std over all latent tokens under `data_dir`.

    Uses Welford-style two-pass aggregation so we don't have to fit all data
    in memory (each shard is loaded one-at-a-time).
    """
    data_dir = Path(data_dir)
    if shard_paths is None:
        shard_paths = _list_shards(data_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No latent shards under {data_dir} to compute stats from.")

    # Pass 1: total count + sum (for mean).
    total = 0
    s = torch.zeros(CODEC_LATENT_DIM, dtype=torch.float64)
    for p in shard_paths:
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        lat = obj["latent"].reshape(-1, obj["latent"].shape[-1]).double()
        s += lat.sum(dim=0)
        total += lat.shape[0]
    mean = (s / total).float()

    # Pass 2: variance (Welford-equivalent two-pass).
    sq = torch.zeros(CODEC_LATENT_DIM, dtype=torch.float64)
    for p in shard_paths:
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        lat = obj["latent"].reshape(-1, obj["latent"].shape[-1]).double()
        sq += (lat - mean.double()).pow(2).sum(dim=0)
    var = sq / max(total - 1, 1)
    std = var.sqrt().float().clamp(min=1e-3)  # guard against degenerate channels

    return LatStats(mean=mean, std=std, count=total)


def stats_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "lat_stats.pt"


def load_or_compute(data_dir: str | Path, shard_paths: List[Path] | None = None) -> LatStats:
    """Load `lat_stats.pt` if present, else compute and cache it."""
    p = stats_path(data_dir)
    if p.exists():
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
            stats = LatStats.from_dict(obj)
            logger.info(
                "Loaded latent stats from %s (count=%d, |mean|=%.4f, mean(std)=%.4f)",
                p, stats.count, stats.mean.abs().mean().item(), stats.std.mean().item(),
            )
            return stats
        except Exception as e:
            logger.warning("Failed to load %s (%s); recomputing.", p, e)
    logger.info("Computing latent stats over %s ...", data_dir)
    stats = compute_lat_stats(data_dir, shard_paths=shard_paths)
    torch.save(stats.to_dict(), str(p))
    logger.info(
        "Saved latent stats to %s (count=%d, |mean|=%.4f, mean(std)=%.4f)",
        p, stats.count, stats.mean.abs().mean().item(), stats.std.mean().item(),
    )
    return stats
