"""Offline sampling: prompt audio -> continuation -> wav.

Usage:
    python -m flow.sample \
        --ckpt        ./runs/v0/ema.pt \
        --prompt-wav  ./prompt.wav     \
        --duration-s  20               \
        --nfe         8                \
        --solver      heun             \
        --out         ./out.wav        \
        --device      mps

If `--prompt-wav` is omitted, the model generates unconditionally (no prefix).
The output `.wav` contains [prompt + generation] concatenated, so you can hear
the seam.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from .codec_wrapper import CodecConfig, CodecWrapper
from .config import (
    CODEC_CHUNK_SAMPLES,
    CODEC_LATENT_DIM,
    CODEC_TOKENS_PER_CHUNK,
    CFMConfig,
    ModelConfig,
)
from .model.cfm import SUPPORTED_SCHEDULES, SUPPORTED_SOLVERS, sample as cfm_sample
from .model.dit import FlowDiT
from .model.ema import EMA
from .utils import best_device, get_logger

logger = get_logger("flow.sample")


def load_model(ckpt_path: str, device: torch.device) -> FlowDiT:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = sd["config"]["model"]
    model_cfg = ModelConfig(**cfg_dict)
    model = FlowDiT(
        latent_dim=model_cfg.latent_dim,
        block_size=model_cfg.block_size,
        dim=model_cfg.dim,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        head_dim=model_cfg.head_dim,
        mlp_mult=model_cfg.mlp_mult,
        cond_dim=model_cfg.cond_dim,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
    )
    # Load EMA weights into the model.
    ema_state = sd.get("ema", None)
    if ema_state is None:
        raise RuntimeError(f"checkpoint {ckpt_path} has no 'ema' field")
    ema = EMA(model, decay=0.0)  # decay value doesn't matter for loading
    ema.load_state_dict(ema_state)
    ema.copy_to(model)
    model.to(device).eval()
    return model


def duration_to_target_tokens(duration_s: float, block_size: int) -> int:
    """Round up to a whole number of chunks."""
    chunks = math.ceil(duration_s * 48_000 / CODEC_CHUNK_SAMPLES)
    return chunks * block_size


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="Path to ema.pt produced by training.")
    p.add_argument("--prompt-wav", default=None, help="Optional prompt audio (any sr, stereo or mono).")
    p.add_argument("--duration-s", type=float, default=20.0,
                   help="Duration of the *generated continuation* in seconds (excludes prompt).")
    p.add_argument("--nfe", type=int, default=8, help="Number of ODE/SDE steps.")
    p.add_argument("--solver", default="heun", choices=list(SUPPORTED_SOLVERS),
                   help="Sampler: euler/heun/midpoint/rk4 (deterministic ODE), "
                        "dpmpp (DPM-Solver++ 2M for RF), pingpong (stochastic SDE).")
    p.add_argument("--schedule", default="linear", choices=list(SUPPORTED_SCHEDULES),
                   help="Time grid: 'linear' (default) or 'shifted' (logSNR shift).")
    p.add_argument("--schedule-shift", type=float, default=0.0,
                   help="LogSNR shift exponent for --schedule shifted (e.g. 1.0-3.0 for long sequences).")
    p.add_argument("--out", required=True, help="Output .wav path.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decode-mode", default="parallel", choices=["parallel", "autoregressive"])
    args = p.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)

    device = best_device(args.device)
    logger.info("device=%s", device)

    # Load model + codec.
    model = load_model(args.ckpt, device)
    block_size = model.block_size
    codec = CodecWrapper(CodecConfig(device=str(device), decode_mode=args.decode_mode))

    # Encode the prompt (optional).
    prefix_latents: Optional[torch.Tensor]
    if args.prompt_wav is not None:
        wv, sr = sf.read(args.prompt_wav, dtype="float32", always_2d=True)
        wv = np.transpose(wv, (1, 0))  # [C, N]
        if wv.shape[0] == 1:
            wv = np.repeat(wv, 2, axis=0)
        elif wv.shape[0] > 2:
            wv = wv[:2]
        wv_t = torch.from_numpy(wv)
        latent = codec.encode_audio(wv_t, sr=sr)  # [T*8, 64] or [T, 8, 64]
        if latent.dim() == 3:
            latent = latent.reshape(-1, latent.shape[-1])
        prefix_latents = latent.unsqueeze(0).to(device)  # [1, P, 64]
        logger.info("prompt: %.2fs of audio -> %d prefix tokens", wv.shape[-1] / sr, prefix_latents.shape[1])
    else:
        prefix_latents = torch.zeros(1, 0, model.latent_dim, device=device)
        logger.info("unconditional generation (no prompt)")

    # Pad prefix to a chunk boundary just in case.
    pad_extra = (-prefix_latents.shape[1]) % block_size
    if pad_extra > 0:
        prefix_latents = torch.nn.functional.pad(prefix_latents, (0, 0, 0, pad_extra))
        logger.info("padded prefix by %d tokens to align to chunk boundary", pad_extra)

    n_target_tokens = duration_to_target_tokens(args.duration_s, block_size)
    logger.info(
        "generating %d target tokens (%.2fs) with %s, NFE=%d",
        n_target_tokens,
        n_target_tokens / block_size * (CODEC_CHUNK_SAMPLES / 48_000),
        args.solver,
        args.nfe,
    )

    full = cfm_sample(
        model,
        prefix=prefix_latents,
        n_target_tokens=n_target_tokens,
        n_steps=args.nfe,
        solver=args.solver,
        schedule=args.schedule,
        schedule_shift=args.schedule_shift,
    )  # [1, P + n_target, 64]

    # Decode through the codec.
    full = full.squeeze(0)  # [L, 64]
    P = prefix_latents.shape[1]
    if P > 0:
        # Split prefix from target so we can save them separately if desired.
        target = full[P:]
    else:
        target = full
    # Just decode the entire concatenated latent stream.
    wv_out = codec.decode_latents(full)
    if wv_out.dim() == 3:
        wv_out = wv_out[0]
    wv_np = wv_out.transpose(0, 1).contiguous().cpu().numpy().astype(np.float32)
    sf.write(args.out, wv_np, codec.sample_rate)
    logger.info("wrote %s (%.2fs @ %d Hz)", args.out, wv_np.shape[0] / codec.sample_rate, codec.sample_rate)


if __name__ == "__main__":
    main()
