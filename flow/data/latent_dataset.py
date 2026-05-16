"""Streaming dataset over pre-encoded latent shards.

Each shard is a `.pt` file written by `flow/data/preencode.py` containing a
`latent` tensor of shape `[T, 8, 64]` and metadata. At training time we serve
random crops of length `crop_tokens` (which must be a multiple of
`block_size`) flattened to `[crop_tokens, 64]`.

The prefix/target split for continuation training is performed *outside* the
dataset (in the train loop) so that we can re-randomize it every step without
re-reading the disk.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from ..config import CODEC_TOKENS_PER_CHUNK
from ..utils import get_logger

logger = get_logger("flow.data.latent_dataset")


@dataclass
class _ShardMeta:
    path: Path
    n_tokens: int  # T * 8


def _list_shards(root: Path) -> List[Path]:
    shards: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.endswith(".pt"):
                continue
            # Skip the latent-stats sidecar (also a .pt file but not a shard).
            if f == "lat_stats.pt":
                continue
            shards.append(Path(dirpath) / f)
    shards.sort()
    return shards


def _scan_shard_meta(path: Path) -> Optional[_ShardMeta]:
    try:
        # mmap=True keeps memory usage low; we only need the latent shape.
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("skip unreadable shard %s: %s", path, e)
        return None
    lat = obj.get("latent", None)
    if lat is None:
        logger.warning("skip shard with no 'latent' field: %s", path)
        return None
    if lat.dim() != 3 or lat.shape[1] != CODEC_TOKENS_PER_CHUNK:
        logger.warning("skip shard with bad shape %s: %s", lat.shape, path)
        return None
    return _ShardMeta(path=path, n_tokens=lat.shape[0] * lat.shape[1])


class LatentDataset(Dataset):
    """Random-crop dataset over a directory of latent shards.

    Args:
        root: directory of .pt shards (searched recursively).
        crop_tokens: length of each crop, in tokens. Must be a multiple of 8.
        seed: RNG seed.
        split: 'train' | 'val' | 'all'.
        val_frac: fraction of shards held out for validation when split != 'all'.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        crop_tokens: int,
        seed: int = 42,
        split: str = "train",
        val_frac: float = 0.05,
    ):
        self.root = Path(root)
        assert crop_tokens % CODEC_TOKENS_PER_CHUNK == 0, (
            f"crop_tokens={crop_tokens} must be a multiple of {CODEC_TOKENS_PER_CHUNK}."
        )
        self.crop_tokens = crop_tokens
        self.split = split

        all_shards = _list_shards(self.root)
        if not all_shards:
            raise FileNotFoundError(f"No .pt shards found under {self.root}")

        # Stable train/val split by hashing path (keeps it identical across runs).
        rng = random.Random(seed)
        all_shards = list(all_shards)
        rng.shuffle(all_shards)
        n_val = max(1, int(len(all_shards) * val_frac)) if val_frac > 0 else 0
        if split == "val":
            picked = all_shards[:n_val]
        elif split == "train":
            picked = all_shards[n_val:]
        else:  # 'all'
            picked = all_shards

        # Filter to shards long enough to be cropped at training length.
        kept: List[_ShardMeta] = []
        for p in picked:
            meta = _scan_shard_meta(p)
            if meta is None:
                continue
            if meta.n_tokens >= crop_tokens:
                kept.append(meta)
            else:
                logger.debug("skip too-short shard %s (%d < %d tokens)", p, meta.n_tokens, crop_tokens)

        if not kept:
            raise RuntimeError(
                f"No shards long enough for crop_tokens={crop_tokens} under {self.root}."
            )

        self.shards = kept
        logger.info(
            "LatentDataset[%s]: %d shards (%.1f h of audio approx)",
            split,
            len(self.shards),
            sum(s.n_tokens for s in self.shards) / (CODEC_TOKENS_PER_CHUNK * 11.72) / 3600,
        )

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, idx: int) -> torch.Tensor:
        meta = self.shards[idx]
        obj = torch.load(str(meta.path), map_location="cpu", weights_only=False)
        latent = obj["latent"]  # [T, 8, 64]
        flat = latent.reshape(-1, latent.shape[-1])  # [T*8, 64]
        n_tokens = flat.shape[0]

        # Pick a crop start aligned to a chunk boundary so that the block
        # structure is preserved.
        max_start = n_tokens - self.crop_tokens
        # Align to multiples of block_size to keep chunk boundaries.
        n_chunks_max = max_start // CODEC_TOKENS_PER_CHUNK
        start_chunk = random.randint(0, n_chunks_max)
        start = start_chunk * CODEC_TOKENS_PER_CHUNK
        crop = flat[start : start + self.crop_tokens]
        return crop.float().contiguous()
