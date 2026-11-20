"""Pre-encode an audio directory into per-file latent shards.

Encoding through CoDiCodec is the slowest part of training when run on the
fly, especially on Apple Silicon where the codec's `torch.compile`'d fast path
is unavailable. We therefore run encoding once and serialize the resulting
continuous latents to disk.

Each output file is a torch `.pt` containing:
    {
        'latent':       FloatTensor [T, 8, 64],   # post-atanh / sigma_rescale
        'sample_rate':  int,                       # the codec rate (48000)
        'samples':      int,                       # waveform length at the codec rate
        'source':       str,                       # absolute source path
    }

Usage:
    python -m flow.data.preencode \
        --in-dir   ~/music/training \
        --out-dir  ./data/latents   \
        --max-seconds 300

Files longer than `--max-seconds` are encoded in segments of whole chunks so
peak memory stays bounded; the result is identical to encoding the whole file.

Augmentation (small datasets): `--offsets N` also encodes each file shifted by
k/N of a chunk (k = 1..N-1), which gives different latents for the same music;
`--swap-lr` adds left/right-swapped copies. Extra variants are written next to
the original as `<name>_o<k>[_lr].pt` and share its `source`, so the train/val
split keeps them together.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from ..codec_wrapper import CodecConfig, CodecWrapper
from ..utils import ensure_dir, get_logger

logger = get_logger("flow.data.preencode")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aif", ".aiff", ".opus", ".wv"}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def find_audio_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if Path(f).suffix.lower() in AUDIO_EXTS:
                out.append(Path(dirpath) / f)
    out.sort()
    return out


def output_path_for(src: Path, in_root: Path, out_root: Path) -> Path:
    rel = src.relative_to(in_root).with_suffix(".pt")
    return out_root / rel


def load_audio_stereo(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file, return float32 stereo array [2, N] and sample rate."""
    wv, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wv = np.transpose(wv, (1, 0))  # [C, N]
    if wv.shape[0] == 1:
        wv = np.repeat(wv, 2, axis=0)
    elif wv.shape[0] > 2:
        wv = wv[:2]
    return wv, sr


# --------------------------------------------------------------------------- #
# Main encoding routine
# --------------------------------------------------------------------------- #

@torch.no_grad()
def encode_wave(codec: CodecWrapper, wv: np.ndarray, max_seconds: float | None = None) -> torch.Tensor:
    """Stereo float32 `[2, N]` at the codec rate -> latents `[T, 8, 64]`.

    Segments are whole chunks plus `codec.frame_overlap` extra samples, so each
    segment's STFT frames line up with whole-file encoding (no zero-padded gap
    at the boundaries). A trailing remainder shorter than one chunk is dropped.
    """
    spc = codec.samples_per_chunk
    n = wv.shape[-1]
    seg = n if not max_seconds or max_seconds <= 0 else max(1, int(max_seconds * codec.sample_rate) // spc) * spc
    pieces: List[torch.Tensor] = []
    for start in range(0, n, seg):
        part = wv[..., start : start + seg + codec.frame_overlap]
        if part.shape[-1] < spc:
            break
        latent = codec.encode_audio(torch.from_numpy(np.ascontiguousarray(part)))
        # The codec returns [num_chunks, tokens_per_chunk, desired_channels] for
        # a single stereo input; collapse any extra leading dims defensively.
        pieces.append(latent.reshape(-1, codec.tokens_per_chunk, codec.cfg.desired_channels).cpu().float())
    return torch.cat(pieces, dim=0)


def variant_paths(dst: Path, offsets: int, swap_lr: bool) -> list:
    """[(offset_index, swap, path)]: the plain encoding keeps the original name."""
    out = []
    for swap in ((False, True) if swap_lr else (False,)):
        for k in range(offsets):
            suffix = (f"_o{k}" if k or swap else "") + ("_lr" if swap else "")
            out.append((k, swap, dst.with_name(dst.stem + suffix + ".pt")))
    return out


@torch.no_grad()
def encode_file(codec: CodecWrapper, src: Path, dst: Path, max_seconds: float | None, overwrite: bool,
                offsets: int = 1, swap_lr: bool = False) -> bool:
    todo = [v for v in variant_paths(dst, offsets, swap_lr) if overwrite or not v[2].exists()]
    if not todo:
        return False
    ensure_dir(str(dst.parent))

    try:
        wv, sr = load_audio_stereo(src)
    except Exception as e:
        logger.warning("skip (load fail) %s: %s", src, e)
        return False
    # Resample first so segments can be cut on the codec's chunk grid.
    if sr != codec.sample_rate:
        wv = codec._resample(wv, sr, codec.sample_rate).numpy()

    # If too short to produce even a single chunk, skip.
    spc = codec.samples_per_chunk
    if wv.shape[-1] < spc:
        logger.info("skip (too short %.2fs < %.2fs) %s", wv.shape[-1] / codec.sample_rate, spc / codec.sample_rate, src)
        return False

    for k, swap, path in todo:
        v = wv[::-1] if swap else wv
        v = v[..., k * spc // offsets:]
        if v.shape[-1] < spc:
            continue
        torch.save(
            {
                "latent": encode_wave(codec, v, max_seconds),  # [T, 8, 64]
                "sample_rate": codec.sample_rate,
                "samples": int(v.shape[-1]),
                "source": str(src.resolve()),
            },
            str(path),
        )
    return True


def run(in_dir: str, out_dir: str, device: str | None, max_seconds: float | None, overwrite: bool,
        offsets: int = 1, swap_lr: bool = False) -> None:
    in_root = Path(in_dir).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    ensure_dir(str(out_root))

    files = find_audio_files(in_root)
    if not files:
        logger.error("No audio files found under %s", in_root)
        return
    logger.info("Found %d audio files under %s", len(files), in_root)

    codec = CodecWrapper(CodecConfig(device=device))

    written = 0
    skipped = 0
    for src in tqdm(files, desc="encoding"):
        dst = output_path_for(src, in_root, out_root)
        ok = encode_file(codec, src, dst, max_seconds=max_seconds, overwrite=overwrite,
                         offsets=offsets, swap_lr=swap_lr)
        written += int(ok)
        skipped += int(not ok)

    logger.info("Done. wrote=%d skipped=%d total=%d -> %s", written, skipped, len(files), out_root)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in-dir", required=True, help="Directory of audio files (recursive).")
    p.add_argument("--out-dir", required=True, help="Where to write per-file .pt latents.")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto).")
    p.add_argument("--max-seconds", type=float, default=300.0,
                   help="Encode files longer than this in segments (bounds memory; same result). <=0 disables.")
    p.add_argument("--overwrite", action="store_true", help="Re-encode files even if the .pt already exists.")
    p.add_argument("--offsets", type=int, default=1,
                   help="Augmentation: also encode each file shifted by k/N of a chunk, k=1..N-1 (N x data).")
    p.add_argument("--swap-lr", action="store_true",
                   help="Augmentation: also encode left/right-swapped copies (2x data).")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        device=args.device,
        max_seconds=args.max_seconds if args.max_seconds and args.max_seconds > 0 else None,
        overwrite=args.overwrite,
        offsets=max(1, args.offsets),
        swap_lr=args.swap_lr,
    )


if __name__ == "__main__":
    main()
