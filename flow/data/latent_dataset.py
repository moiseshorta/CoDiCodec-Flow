"""In-memory dataset over pre-encoded latent shards.

Each shard is a `.pt` file written by `flow/data/preencode.py` containing a
`latent` tensor of shape `[T, 8, 64]` and its `source` audio path. Latents are
tiny (1 h of audio ~ 11 MB), so a whole split is loaded once into one tensor
on the training device and crops are gathered there: no workers, no disk reads
in the training loop.

Crops of `crop_tokens` (a multiple of 8) are drawn uniformly over every valid
chunk-aligned start, so each second of audio is equally likely regardless of
how it is split into files. The train/val split is by source recording, so
augmented variants of one file (`preencode --offsets/--swap-lr`) never
straddle it.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List

import torch

from ..config import CODEC_FRAME_RATE_HZ, CODEC_TOKENS_PER_CHUNK
from ..utils import get_logger

logger = get_logger("flow.data.latent_dataset")


def _list_shards(root: Path) -> List[Path]:
    shards: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            # Skip the old latent-stats sidecar (a .pt file but not a shard).
            if f.endswith(".pt") and f != "lat_stats.pt":
                shards.append(Path(dirpath) / f)
    shards.sort()
    return shards


class LatentDataset:
    """One split: latent pieces concatenated on `device`, sampled as random crops."""

    def __init__(self, pieces: List[torch.Tensor], crop_tokens: int, device, name: str):
        assert crop_tokens % CODEC_TOKENS_PER_CHUNK == 0, (
            f"crop_tokens={crop_tokens} must be a multiple of {CODEC_TOKENS_PER_CHUNK}."
        )
        kept = [p for p in pieces if p.shape[0] >= crop_tokens]
        dropped = sum(p.shape[0] for p in pieces if p.shape[0] < crop_tokens)
        if not kept:
            raise RuntimeError(
                f"No {name} data: every piece is shorter than crop_tokens={crop_tokens} "
                f"(~{crop_tokens / CODEC_FRAME_RATE_HZ:.0f} s). Use longer files or a smaller --crop-tokens."
            )
        data = torch.cat(kept)
        starts, offset = [], 0
        for p in kept:
            starts.append(offset + torch.arange(0, p.shape[0] - crop_tokens + 1, CODEC_TOKENS_PER_CHUNK))
            offset += p.shape[0]
        # Per-channel stats of the data actually trained on (float64 on CPU: MPS has no fp64).
        std, mean = torch.std_mean(data.double(), dim=0)
        self.mean, self.std = mean.float(), std.float()
        self.data = data.to(device)
        self.starts = torch.cat(starts).to(device)
        self.crop_tokens = crop_tokens
        self._offsets = torch.arange(crop_tokens, device=device)
        logger.info(
            "LatentDataset[%s]: %d pieces, %.2f h, %d crop positions%s",
            name, len(kept), data.shape[0] / CODEC_FRAME_RATE_HZ / 3600, len(self.starts),
            f"; dropped {len(pieces) - len(kept)} pieces shorter than a crop "
            f"({dropped / CODEC_FRAME_RATE_HZ / 3600:.2f} h)" if dropped else "",
        )

    def __len__(self) -> int:
        return len(self.starts)

    def _crops(self, idx: torch.Tensor) -> torch.Tensor:
        return self.data[self.starts[idx].unsqueeze(1) + self._offsets]  # [B, L, D]

    def sample(self, batch_size: int) -> torch.Tensor:
        """`batch_size` random crops `[B, crop_tokens, D]` (raw codec space)."""
        return self._crops(torch.randint(len(self.starts), (batch_size,), device=self.starts.device))

    def fixed_crops(self, n: int) -> torch.Tensor:
        """`n` evenly spaced crops: deterministic validation / preview prompts."""
        return self._crops(torch.linspace(0, len(self.starts) - 1, n, device=self.starts.device).long())


def load_splits(root, crop_tokens: int, val_frac: float = 0.05, seed: int = 42, device="cpu"):
    """Load every shard under `root` once; return `(train, val)` split by source recording."""
    by_source: dict = {}
    for path in _list_shards(Path(root)):
        try:
            obj = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning("skip unreadable shard %s: %s", path, e)
            continue
        lat = obj.get("latent") if isinstance(obj, dict) else None
        if lat is None or lat.dim() != 3 or lat.shape[1] != CODEC_TOKENS_PER_CHUNK:
            logger.warning("skip shard without a [T, %d, D] 'latent': %s", CODEC_TOKENS_PER_CHUNK, path)
            continue
        by_source.setdefault(obj.get("source", str(path)), []).append(lat.reshape(-1, lat.shape[-1]).float())
    if not by_source:
        raise FileNotFoundError(f"No latent shards found under {root}")

    sources = sorted(by_source)
    random.Random(seed).shuffle(sources)
    n_val = max(1, int(len(sources) * val_frac)) if val_frac > 0 else 0
    if n_val >= len(sources):
        raise RuntimeError(f"Need at least 2 source recordings for a train/val split; found {len(sources)}.")
    pieces = lambda srcs: [p for s in srcs for p in by_source[s]]
    train = LatentDataset(pieces(sources[n_val:]), crop_tokens, device, "train")
    val = LatentDataset(pieces(sources[:n_val]), crop_tokens, device, "val") if n_val else None
    return train, val
