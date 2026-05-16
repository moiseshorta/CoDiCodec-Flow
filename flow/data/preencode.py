"""Pre-encode an audio directory into per-file latent shards.

Encoding through CoDiCodec is the slowest part of training when run on the
fly, especially on Apple Silicon where the codec's `torch.compile`'d fast path
is unavailable. We therefore run encoding once and serialize the resulting
continuous latents to disk.

Each output file is a torch `.pt` containing:
    {
        'latent':       FloatTensor [T, 8, 64],   # post-atanh / sigma_rescale
        'sample_rate':  int,                       # the codec rate (48000)
        'samples':      int,                       # original waveform length
        'source':       str,                       # absolute source path
    }

Usage:
    python -m flow.data.preencode \
        --in-dir   ~/music/training \
        --out-dir  ./data/latents   \
        --device   mps              \
        --max-seconds 60

If `--max-seconds` is set, files longer than this are split into consecutive
chunks before encoding so peak memory stays bounded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

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


def split_into_segments(wv: np.ndarray, sr: int, max_seconds: float | None) -> Iterable[np.ndarray]:
    if max_seconds is None or max_seconds <= 0:
        yield wv
        return
    seg_len = int(max_seconds * sr)
    n = wv.shape[-1]
    for start in range(0, n, seg_len):
        yield wv[..., start : start + seg_len]


# --------------------------------------------------------------------------- #
# Main encoding routine
# --------------------------------------------------------------------------- #

@torch.no_grad()
def encode_file(codec: CodecWrapper, src: Path, dst: Path, max_seconds: float | None, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    ensure_dir(str(dst.parent))

    try:
        wv, sr = load_audio_stereo(src)
    except Exception as e:
        logger.warning("skip (load fail) %s: %s", src, e)
        return False

    # If too short to produce even a single chunk, skip.
    min_samples = codec.samples_per_chunk
    if wv.shape[-1] < min_samples:
        logger.info("skip (too short %ds < %.2fs) %s", wv.shape[-1], min_samples / codec.sample_rate, src)
        return False

    pieces: List[torch.Tensor] = []
    for seg in split_into_segments(wv, sr, max_seconds):
        if seg.shape[-1] < min_samples:
            continue
        seg_t = torch.from_numpy(seg)
        latent = codec.encode_audio(seg_t, sr=sr)
        # The codec returns [num_chunks, tokens_per_chunk, desired_channels] for
        # a single stereo input; collapse any extra leading dims defensively.
        latent = latent.reshape(-1, codec.tokens_per_chunk, codec.cfg.desired_channels)
        pieces.append(latent.cpu().to(torch.float32))

    if not pieces:
        return False

    latent_full = torch.cat(pieces, dim=0)  # [T, 8, 64]
    torch.save(
        {
            "latent": latent_full,
            "sample_rate": codec.sample_rate,
            "samples": int(wv.shape[-1]),
            "source": str(src.resolve()),
        },
        str(dst),
    )
    return True


def run(in_dir: str, out_dir: str, device: str | None, max_seconds: float | None, overwrite: bool) -> None:
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
        ok = encode_file(codec, src, dst, max_seconds=max_seconds, overwrite=overwrite)
        written += int(ok)
        skipped += int(not ok)

    logger.info("Done. wrote=%d skipped=%d total=%d -> %s", written, skipped, len(files), out_root)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in-dir", required=True, help="Directory of audio files (recursive).")
    p.add_argument("--out-dir", required=True, help="Where to write per-file .pt latents.")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto).")
    p.add_argument("--max-seconds", type=float, default=60.0,
                   help="Split files longer than this into segments before encoding. <=0 disables.")
    p.add_argument("--overwrite", action="store_true", help="Re-encode files even if the .pt already exists.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    run(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        device=args.device,
        max_seconds=args.max_seconds if args.max_seconds and args.max_seconds > 0 else None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
