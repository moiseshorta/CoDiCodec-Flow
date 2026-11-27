"""Offline sampling: prompt audio -> continuation -> wav.

Usage:
    python -m flow.sample \
        --ckpt        ./runs/v0/ema.pt \
        --prompt-wav  ./prompt.wav     \
        --duration-s  20               \
        --nfe         8                \
        --solver      heun             \
        --out         ./out.wav

If `--prompt-wav` is omitted, the model generates unconditionally (no prefix).
`--mode joint` (default) denoises the whole continuation at once; prompt +
continuation must fit the model window (~87 s). `--mode chunked` generates
chunk by chunk with a sliding context window, exactly like `realtime`, so any
duration works. The output `.wav` contains [prompt + generation] concatenated,
so you can hear the seam.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict

import numpy as np
import soundfile as sf
import torch

from .codec_wrapper import CodecConfig, CodecWrapper
from .config import CODEC_CHUNK_SAMPLES, CODEC_SAMPLE_RATE, ModelConfig
from .model.cfm import SUPPORTED_SCHEDULES, SUPPORTED_SOLVERS, generate, sample_joint
from .model.dit import FlowDiT
from .lora import LoraBank
from .model.ema import EMA
from .utils import best_device, get_logger

logger = get_logger("flow.sample")


def load_model(ckpt_path: str, device: torch.device, use_ema: bool = True) -> FlowDiT:
    """Rebuild a FlowDiT from `ema.pt` or `last.pt`; EMA weights unless `use_ema=False`."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "config" not in sd or "model" not in sd["config"]:
        raise RuntimeError(f"checkpoint {ckpt_path} has no 'config.model' field; cannot rebuild architecture")
    model = FlowDiT(**{**asdict(ModelConfig(**sd["config"]["model"])), "dropout": 0.0})
    if use_ema or "model" not in sd:  # ema.pt holds only EMA weights: no --use-ema needed
        if sd.get("ema") is None:
            raise RuntimeError(f"checkpoint {ckpt_path} has no 'ema' field")
        ema = EMA(model, decay=0.0)  # decay value doesn't matter for loading
        ema.load_state_dict(sd["ema"])
        ema.copy_to(model)
        logger.info("Loaded EMA weights from %s", ckpt_path)
    else:
        if "model" not in sd:
            raise RuntimeError(f"checkpoint {ckpt_path} has no 'model' field (ema.pt only has EMA weights)")
        model.load_state_dict(sd["model"])
        logger.info("Loaded raw (non-EMA) weights from %s", ckpt_path)
    return model.to(device).eval()


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="Path to ema.pt produced by training.")
    p.add_argument("--lora", nargs="+", default=None, help="LoRA files over --ckpt, applied together at strength 1.")
    p.add_argument("--prompt-wav", default=None, help="Optional prompt audio (any sr, stereo or mono).")
    p.add_argument("--duration-s", type=float, default=20.0,
                   help="Duration of the *generated continuation* in seconds (excludes prompt).")
    p.add_argument("--nfe", type=int, default=8,
                   help="Solver steps per chunk (model calls per step: heun/midpoint 2, rk4 4, others 1).")
    p.add_argument("--solver", default="heun", choices=list(SUPPORTED_SOLVERS),
                   help="Sampler: euler/heun/midpoint/rk4 (deterministic ODE), "
                        "dpmpp (DPM-Solver++ 2M for RF), pingpong (stochastic SDE).")
    p.add_argument("--schedule", default="linear", choices=list(SUPPORTED_SCHEDULES),
                   help="Time grid: 'linear' (default) or 'shifted' (logSNR shift).")
    p.add_argument("--schedule-shift", type=float, default=0.0,
                   help="LogSNR shift exponent for --schedule shifted (e.g. 1.0-3.0).")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Velocity scaling: <1 sharpens, >1 leaves more noise.")
    p.add_argument("--mode", default="joint", choices=["joint", "chunked"],
                   help="joint: denoise the whole continuation at once (fits ~87 s with the prompt). "
                        "chunked: chunk by chunk like realtime, any duration.")
    p.add_argument("--context-chunks", type=int, default=32,
                   help="--mode chunked: sliding context window in chunks (1 chunk ~ 0.683 s).")
    p.add_argument("--out", required=True, help="Output .wav path.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decode-mode", default="parallel", choices=["parallel", "autoregressive"])
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = best_device(args.device)
    logger.info("device=%s", device)

    # Load model + codec.
    model = load_model(args.ckpt, device)
    if args.lora:
        bank = LoraBank(model, args.lora)
        bank.set({name: 1.0 for name in bank.loras})
    codec = CodecWrapper(CodecConfig(device=str(device), decode_mode=args.decode_mode))

    # Encode the prompt (optional).
    if args.prompt_wav is not None:
        wv, sr = sf.read(args.prompt_wav, dtype="float32", always_2d=True)
        wv = np.transpose(wv, (1, 0))  # [C, N]
        if wv.shape[0] == 1:
            wv = np.repeat(wv, 2, axis=0)
        elif wv.shape[0] > 2:
            wv = wv[:2]
        latent = codec.encode_audio(torch.from_numpy(wv), sr=sr)  # [T, 8, 64]
        prefix = latent.reshape(-1, latent.shape[-1]).unsqueeze(0).to(device)  # [1, P, 64]
        logger.info("prompt: %.2fs of audio -> %d prefix tokens", wv.shape[-1] / sr, prefix.shape[1])
    else:
        prefix = torch.zeros(1, 0, model.latent_dim, device=device)
        logger.info("unconditional generation (no prompt)")

    n_chunks = math.ceil(args.duration_s * CODEC_SAMPLE_RATE / CODEC_CHUNK_SAMPLES)
    n_tokens = n_chunks * model.block_size
    if args.mode == "joint" and prefix.shape[1] + n_tokens > model.max_seq_len:
        window_s = model.max_seq_len / model.block_size * CODEC_CHUNK_SAMPLES / CODEC_SAMPLE_RATE
        raise SystemExit(f"prompt + {args.duration_s:.0f} s exceeds the {window_s:.0f} s joint window; "
                         "shorten it or use --mode chunked")
    logger.info("generating %d chunks (%.2fs) with %s, %d steps, mode=%s",
                n_chunks, n_chunks * CODEC_CHUNK_SAMPLES / CODEC_SAMPLE_RATE, args.solver, args.nfe, args.mode)
    kw = dict(n_steps=args.nfe, solver=args.solver, temperature=args.temperature,
              schedule=args.schedule, schedule_shift=args.schedule_shift)
    if args.mode == "joint":
        full = sample_joint(model, prefix, n_tokens, **kw)
    else:
        full = generate(model, prefix, n_chunks, context_chunks=args.context_chunks, **kw)
    # full: [1, P + n_chunks * 8, 64]

    # Decode the entire concatenated latent stream.
    wv_out = codec.decode_latents(full.squeeze(0))
    if wv_out.dim() == 3:
        wv_out = wv_out[0]
    wv_np = wv_out.transpose(0, 1).contiguous().cpu().numpy().astype(np.float32)
    sf.write(args.out, wv_np, codec.sample_rate)
    logger.info("wrote %s (%.2fs @ %d Hz)", args.out, wv_np.shape[0] / codec.sample_rate, codec.sample_rate)


if __name__ == "__main__":
    main()
