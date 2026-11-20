"""Training loop for the block-causal Flow-Matching DiT.

This script is intentionally dependency-light (no hydra, no wandb): plain
argparse + dataclasses + tqdm. Designed to be runnable on a single GPU (CUDA
or MPS).

Usage:
    python -m flow.train \
        --data-dir ./data/latents \
        --out-dir  ./runs/v0      \
        --batch-size 8 --max-steps 200000

Checkpoints are written every `--ckpt-every` steps:
    {out_dir}/last.pt   - model + optim + EMA, for resuming
    {out_dir}/ema.pt    - EMA-only state_dict, for inference
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .config import Config, ModelConfig
from .data.latent_dataset import LatentDataset, load_splits
from .model.cfm import (
    SUPPORTED_SOLVERS,
    cfm_loss,
    generate,
    sample_joint,
    sample_prefix_chunks,
    sample_t,
    teacher_forcing_loss,
)
from .model.dit import FlowDiT
from .model.ema import EMA
from .utils import (
    autocast_ctx,
    best_device,
    block_causal_mask,
    count_params,
    ensure_dir,
    get_logger,
    human_int,
    set_seed,
    teacher_forcing_mask,
)


logger = get_logger("flow.train")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def build_model(cfg: ModelConfig, lat_mean: Optional[torch.Tensor] = None,
                lat_std: Optional[torch.Tensor] = None) -> FlowDiT:
    """Construct a FlowDiT, optionally baking in dataset latent normalization."""
    return FlowDiT(**asdict(cfg), lat_mean=lat_mean, lat_std=lat_std)


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "fp32":
        return torch.float32
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    raise ValueError(f"unknown dtype: {s}")


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Train loop
# --------------------------------------------------------------------------- #

def train(cfg: Config) -> None:
    set_seed(cfg.train.seed)
    device = best_device(cfg.device)
    dtype = parse_dtype(cfg.train.dtype)
    teacher_forcing = cfg.cfm.objective == "teacher_forcing"
    if cfg.cfm.objective not in ("teacher_forcing", "joint"):
        raise ValueError(f"unknown objective: {cfg.cfm.objective!r}")
    logger.info("device=%s dtype=%s objective=%s", device, dtype, cfg.cfm.objective)

    out_dir = ensure_dir(cfg.train.out_dir)
    L = cfg.data.crop_tokens
    block_size = cfg.model.block_size
    if L > cfg.model.max_seq_len:
        raise ValueError(f"crop_tokens={L} exceeds max_seq_len={cfg.model.max_seq_len}")

    # ------- data: whole split in memory, crops gathered on device -------- #
    train_ds, val_ds = load_splits(
        cfg.data.data_dir, L, val_frac=cfg.data.val_frac, seed=cfg.data.seed, device=device,
    )

    # ------- model ------------------------------------------------------- #
    # Per-channel normalization (SD3 / Stable Audio 2) so the data matches the
    # N(0, I) noise at t=1; baked in as buffers so checkpoints are self-contained.
    model = build_model(cfg.model, train_ds.mean, train_ds.std).to(device)
    n_params = count_params(model)
    logger.info("Model: %s params (%s)", n_params, human_int(n_params))

    ema = EMA(model, decay=cfg.train.ema_decay)

    # ------- optim ------------------------------------------------------- #
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
        eps=1e-8,
        fused=(device.type == "cuda"),
    )

    # GradScaler is only useful with fp16 on CUDA; bf16 doesn't need it.
    scaler = torch.amp.GradScaler(device.type) if (dtype == torch.float16 and device.type == "cuda") else None

    # Masks for the crop length. Validation always uses the teacher-forcing
    # layout: it measures the generation condition for either objective.
    tf_mask = teacher_forcing_mask(L, block_size, device=device)
    train_mask = tf_mask if teacher_forcing else block_causal_mask(L, block_size, device=device)
    fwd = torch.compile(model) if cfg.train.compile else model

    # ------- resume / init-from ----------------------------------------- #
    start_step = 0
    last_ckpt = Path(out_dir) / "last.pt"

    def _check_compat(sd_model: dict, src: str) -> None:
        """Reject checkpoints from the legacy pre-RoPE / pre-norm architecture."""
        old_keys = {"pe_chunk", "pe_intra"}
        new_keys = {"lat_mean", "lat_std"}
        if any(k in sd_model for k in old_keys) or not any(k in sd_model for k in new_keys):
            raise RuntimeError(
                f"Checkpoint at {src} is from an incompatible architecture "
                "(pre-RoPE / pre-latent-norm). Move it aside or use a new run dir."
            )

    if last_ckpt.exists():
        # Auto-resume: full state restore (model + optim + EMA + step + buffers).
        logger.info("Resuming from %s", last_ckpt)
        sd = torch.load(str(last_ckpt), map_location="cpu", weights_only=True)
        sd_model = sd["model"]
        _check_compat(sd_model, str(last_ckpt))
        model.load_state_dict(sd_model)
        optim.load_state_dict(sd["optim"])
        ema.load_state_dict(sd["ema"])
        start_step = int(sd.get("step", 0))
    elif cfg.train.init_from:
        # Fine-tune workflow: load weights only; keep fresh optimizer, fresh
        # step counter, and -- crucially -- the lat_mean/lat_std buffers we
        # just baked in from the *new* dataset. Source checkpoint's lat_stats
        # are dropped before load.
        init_path = Path(cfg.train.init_from).expanduser().resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from checkpoint not found: {init_path}")
        logger.info("Initializing weights from %s (fine-tune mode)", init_path)
        sd = torch.load(str(init_path), map_location="cpu", weights_only=True)
        if "model" not in sd:
            raise RuntimeError(
                f"Init checkpoint {init_path} has no 'model' state dict; "
                "pass last.pt (not ema.pt) to --init-from."
            )
        sd_model = dict(sd["model"])
        _check_compat(sd_model, str(init_path))
        # Drop dataset-dependent buffers; keep our newly computed ones.
        for k in ("lat_mean", "lat_std"):
            sd_model.pop(k, None)
        missing, unexpected = model.load_state_dict(sd_model, strict=False)
        logger.info(
            "init-from: loaded weights (preserving fresh lat_stats). missing=%d unexpected=%d",
            len(missing), len(unexpected),
        )
        # Seed EMA shadow from the source checkpoint's EMA shadow if available
        # (better starting quality). Refresh buffers from current model so the
        # EMA tracker uses the new dataset's lat_stats.
        if "ema" in sd and isinstance(sd["ema"], dict):
            ema.load_state_dict(sd["ema"])
            for name, b in model.named_buffers():
                ema.buffers[name] = b.detach().clone()
            logger.info("init-from: seeded EMA shadow from source checkpoint")

    # ------- train loop -------------------------------------------------- #
    model.train()
    t0 = time.time()
    accum = 0
    B = cfg.train.batch_size
    n_chunks = L // block_size
    min_snr = cfg.cfm.min_snr_gamma if cfg.cfm.min_snr_gamma and cfg.cfm.min_snr_gamma > 0 else None
    # Summed on device; read back only when logging (no per-step GPU sync).
    running_loss = torch.zeros((), device=device)
    running_n = 0
    # Lazy CodecWrapper holder; only built when audio sampling fires.
    codec_state: dict = {"codec": None}

    # Generate samples at step 0 so we hear what untrained model sounds like.
    if cfg.train.audio_sample_every > 0:
        try:
            generate_audio_samples(
                model, ema, val_ds or train_ds, cfg, device,
                step=start_step, out_dir=out_dir, codec_state=codec_state,
            )
        except Exception as e:
            logger.warning("initial audio sample failed (continuing): %s", e)

    pbar = tqdm(range(start_step, cfg.train.max_steps), initial=start_step,
                total=cfg.train.max_steps, dynamic_ncols=True, desc="train")
    with logging_redirect_tqdm(loggers=[logging.getLogger("flow")]):
        for step in pbar:
            # Cosine LR schedule.
            lr = cosine_lr(step, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.lr)
            for g in optim.param_groups:
                g["lr"] = lr

            x_data = train_ds.sample(B)  # [B, L, D] on device

            # Forward + loss in autocast.
            with autocast_ctx(device, dtype=dtype, enabled=(dtype != torch.float32)):
                if teacher_forcing:
                    t_chunk = sample_t(
                        B * n_chunks, mode=cfg.cfm.t_sample_mode, device=device,
                        logit_normal_loc=cfg.cfm.logit_normal_loc,
                        logit_normal_scale=cfg.cfm.logit_normal_scale,
                    ).view(B, n_chunks)
                    loss = teacher_forcing_loss(fwd, x_data, t_chunk, train_mask, min_snr_gamma=min_snr)
                else:
                    t_sample = sample_t(
                        B, mode=cfg.cfm.t_sample_mode, device=device,
                        logit_normal_loc=cfg.cfm.logit_normal_loc,
                        logit_normal_scale=cfg.cfm.logit_normal_scale,
                    )
                    prefix_len = block_size * sample_prefix_chunks(
                        n_chunks, cfg.cfm.p_unconditional, cfg.cfm.prefix_min_frac, cfg.cfm.prefix_max_frac,
                    )
                    loss = cfm_loss(fwd, x_data, prefix_len, t_sample, attn_mask=train_mask,
                                    min_snr_gamma=min_snr).loss

            if scaler is not None:
                scaler.scale(loss / cfg.train.grad_accum).backward()
            else:
                (loss / cfg.train.grad_accum).backward()

            accum += 1
            running_loss += loss.detach()
            running_n += 1

            if accum >= cfg.train.grad_accum:
                if scaler is not None:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                if scaler is not None:
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                ema.update(model)
                accum = 0

            # Logging
            if (step + 1) % cfg.train.log_every == 0 and running_n > 0:
                avg = float(running_loss / running_n)
                steps_per_sec = (step + 1 - start_step) / max(1e-6, time.time() - t0)
                logger.info(
                    "step=%d/%d  loss=%.4f  lr=%.2e  steps/s=%.2f",
                    step + 1, cfg.train.max_steps, avg, lr, steps_per_sec,
                )
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.1e}")
                running_loss.zero_()
                running_n = 0

            # Validation
            if val_ds is not None and (step + 1) % cfg.train.val_every == 0:
                val_loss = evaluate(model, val_ds, tf_mask, device, dtype)
                logger.info("[val] step=%d  next_chunk_loss=%.4f", step + 1, val_loss)
                model.train()

            # Audio sampling
            if cfg.train.audio_sample_every > 0 and (step + 1) % cfg.train.audio_sample_every == 0:
                try:
                    generate_audio_samples(
                        model, ema, val_ds or train_ds, cfg, device,
                        step=step + 1, out_dir=out_dir, codec_state=codec_state,
                    )
                except Exception as e:
                    logger.warning("audio sample at step %d failed (continuing): %s", step + 1, e)

            # Checkpoint
            if (step + 1) % cfg.train.ckpt_every == 0 or (step + 1) == cfg.train.max_steps:
                save_checkpoint(out_dir, model, optim, ema, cfg, step + 1)

    logger.info("Training done.")


@torch.no_grad()
def generate_audio_samples(
    model: FlowDiT,
    ema: EMA,
    val_ds: LatentDataset,
    cfg: Config,
    device: torch.device,
    step: int,
    out_dir: str,
    codec_state: dict,
) -> None:
    """Generate a few clips and save as .wav: whole-continuation sampling for the
    joint objective, chunk by chunk (exactly as `realtime` does) for teacher forcing.

    Loads the CodecWrapper lazily on first call (downloads ~600 MB on first
    run). Optionally applies EMA weights for sampling (and restores after).
    """
    import soundfile as sf
    import numpy as np

    # Lazy codec construction (so people without it can still run train.py with
    # audio_sample_every=0).
    if codec_state.get("codec") is None:
        from .codec_wrapper import CodecConfig, CodecWrapper
        logger.info("Building CodecWrapper for audio sampling (first call) ...")
        codec_state["codec"] = CodecWrapper(CodecConfig(device=str(device)))
    codec = codec_state["codec"]
    sr = codec.sample_rate
    samples_dir = ensure_dir(os.path.join(out_dir, "samples"))

    # Optional: swap in EMA weights for sampling.
    backup: dict[str, torch.Tensor] = {}
    if cfg.train.audio_use_ema:
        backup = {n: p.detach().clone() for n, p in model.named_parameters()}
        ema.copy_to(model)
    was_training = model.training
    model.eval()

    # Total length is prompt_seconds + continuation_seconds in both modes. In
    # unconditional mode the model generates all of it; in continuation mode
    # the first `prompt_seconds` come from fixed val clips (same prompts every
    # call, so progress is comparable across steps).
    spc = codec.samples_per_chunk
    prompt_chunks = max(1, round(cfg.train.audio_prompt_seconds * sr / spc))
    cont_chunks = max(1, round(cfg.train.audio_continuation_seconds * sr / spc))
    n = max(1, cfg.train.audio_n_samples)
    if cfg.train.audio_unconditional:
        prompt = torch.zeros(n, 0, cfg.model.latent_dim, device=device)
        n_chunks = prompt_chunks + cont_chunks
    else:
        prompt = val_ds.fixed_crops(n)[:, : prompt_chunks * cfg.model.block_size]
        n_chunks = cont_chunks
    mode = "uncond" if cfg.train.audio_unconditional else "cont"

    try:
        if cfg.cfm.objective == "joint":
            full = sample_joint(model, prompt, n_chunks * cfg.model.block_size,
                                n_steps=cfg.train.audio_nfe, solver=cfg.train.audio_solver)
        else:
            full = generate(model, prompt, n_chunks, n_steps=cfg.train.audio_nfe, solver=cfg.train.audio_solver)
        for i in range(n):
            wv = codec.decode_latents(full[i].float())  # [2, N]
            # Clip so garbage at init can't blow out your speakers.
            wv_np = np.clip(wv.transpose(0, 1).cpu().numpy().astype(np.float32), -1.0, 1.0)
            out_path = os.path.join(samples_dir, f"step_{step:07d}_idx_{i:02d}.wav")
            sf.write(out_path, wv_np, sr)
            logger.info("[audio] step=%d idx=%d (%s) -> %s (%.2fs)", step, i, mode, out_path, len(wv_np) / sr)
    finally:
        # Restore non-EMA weights and training mode.
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])
        if was_training:
            model.train()


@torch.no_grad()
def evaluate(model, val_ds: LatentDataset, tf_mask, device, dtype, n_crops: int = 16, seed: int = 0) -> float:
    """Next-chunk validation loss: every chunk denoised from its clean preceding
    chunks (the generation condition), whatever objective is being trained.

    Fixed crops, a fixed grid of noise levels (each chunk sees every level once)
    and seeded noise make the number reproducible across calls and runs.
    """
    model.eval()
    crops = val_ds.fixed_crops(n_crops)  # [N, L, D]
    n_chunks = crops.shape[1] // model.block_size
    t_grid = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total = 0.0
    for k in range(len(t_grid)):
        t_chunk = t_grid[(torch.arange(n_chunks, device=device) + k) % len(t_grid)].expand(len(crops), -1)
        eps = torch.randn(crops.shape, generator=gen).to(device)
        with autocast_ctx(device, dtype=dtype, enabled=(dtype != torch.float32)):
            total += float(teacher_forcing_loss(model, crops, t_chunk, tf_mask, eps=eps))
    return total / len(t_grid)


def _atomic_save(obj, path: Path) -> None:
    # Write then rename: a crash mid-save never leaves a truncated checkpoint.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(tmp, path)


def save_checkpoint(out_dir: str, model, optim, ema: EMA, cfg: Config, step: int) -> None:
    last = Path(out_dir) / "last.pt"
    ema_sd = ema.state_dict()
    _atomic_save(
        {
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "ema": ema_sd,
            "config": {
                "model": asdict(cfg.model),
                "cfm": asdict(cfg.cfm),
                "data": asdict(cfg.data),
                "train": asdict(cfg.train),
                "device": cfg.device,
            },
        },
        last,
    )
    # Standalone EMA file for inference.
    ema_path = Path(out_dir) / "ema.pt"
    _atomic_save(
        {
            "ema": ema_sd,
            "config": {
                "model": asdict(cfg.model),
                "cfm": asdict(cfg.cfm),
            },
            "step": step,
        },
        ema_path,
    )
    logger.info("[ckpt] step=%d  -> %s, %s", step, last.name, ema_path.name)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the codicodec-flow CFM model.")
    p.add_argument("--data-dir", default=None, help="Override DataConfig.data_dir")
    p.add_argument("--out-dir", default=None, help="Override TrainConfig.out_dir")
    p.add_argument("--init-from", default=None,
                   help="Path to last.pt to initialize weights from (fine-tune). "
                        "Optimizer / step counter start fresh; lat_stats stay tied "
                        "to the new dataset. Ignored if out-dir/last.pt exists.")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dtype", default=None, help="fp32 | bf16 | fp16")
    p.add_argument("--crop-tokens", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--compile", action="store_true", default=None,
                   help="torch.compile the training forward (CUDA; ~1 min startup).")
    # Audio sampling
    p.add_argument("--audio-sample-every", type=int, default=None,
                   help="0 disables audio sampling during training.")
    p.add_argument("--audio-n-samples", type=int, default=None)
    p.add_argument("--audio-prompt-seconds", type=float, default=None)
    p.add_argument("--audio-continuation-seconds", type=float, default=None)
    p.add_argument("--audio-nfe", type=int, default=None, help="Solver steps per chunk.")
    p.add_argument("--audio-solver", default=None, choices=list(SUPPORTED_SOLVERS))
    p.add_argument("--audio-unconditional", dest="audio_unconditional", action="store_true", default=None,
                   help="Generate audio samples unconditionally from pure noise (default).")
    p.add_argument("--audio-continuation", dest="audio_unconditional", action="store_false",
                   help="Generate audio samples by continuing a val-set prompt.")
    # CFM / loss knobs.
    p.add_argument("--objective", default=None, choices=["joint", "teacher_forcing"],
                   help="joint (default): clean prefix + a target span denoised jointly. "
                        "teacher_forcing: train every chunk the way realtime generates it; "
                        "learns faster but overfits small datasets sooner (use --offsets).")
    p.add_argument("--t-sample-mode", default=None, choices=["logit_normal", "uniform"],
                   help="Distribution for sampling the noise level t during training.")
    p.add_argument("--min-snr-gamma", type=float, default=None,
                   help="Min-SNR-γ loss weighting (Hang et al. 2023). 0 disables.")
    p.add_argument("--dropout", type=float, default=None,
                   help="Residual / MLP dropout. Raise (0.15-0.2) for tiny datasets.")
    # Model size (overrides ModelConfig defaults).
    p.add_argument("--dim", type=int, default=None, help="Transformer hidden size.")
    p.add_argument("--n-layers", type=int, default=None, help="Number of transformer blocks.")
    p.add_argument("--n-heads", type=int, default=None, help="Number of attention heads.")
    p.add_argument("--head-dim", type=int, default=None, help="Per-head dim. Must satisfy dim = n_heads * head_dim.")
    p.add_argument("--cond-dim", type=int, default=None,
                   help="Time conditioning dim (defaults to --dim if unset).")
    p.add_argument("--mlp-mult", type=int, default=None, help="MLP hidden = dim * mlp_mult (post SwiGLU split).")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = Config()
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.init_from:
        cfg.train.init_from = args.init_from
    if args.device:
        cfg.device = args.device
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.grad_accum:
        cfg.train.grad_accum = args.grad_accum
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    if args.warmup_steps is not None:
        cfg.train.warmup_steps = args.warmup_steps
    if args.lr:
        cfg.train.lr = args.lr
    if args.dtype:
        cfg.train.dtype = args.dtype
    if args.crop_tokens:
        cfg.data.crop_tokens = args.crop_tokens
    if args.log_every:
        cfg.train.log_every = args.log_every
    if args.val_every:
        cfg.train.val_every = args.val_every
    if args.ckpt_every:
        cfg.train.ckpt_every = args.ckpt_every
    if args.compile:
        cfg.train.compile = True
    if args.audio_sample_every is not None:
        cfg.train.audio_sample_every = args.audio_sample_every
    if args.audio_n_samples:
        cfg.train.audio_n_samples = args.audio_n_samples
    if args.audio_prompt_seconds:
        cfg.train.audio_prompt_seconds = args.audio_prompt_seconds
    if args.audio_continuation_seconds:
        cfg.train.audio_continuation_seconds = args.audio_continuation_seconds
    if args.audio_nfe:
        cfg.train.audio_nfe = args.audio_nfe
    if args.audio_solver:
        cfg.train.audio_solver = args.audio_solver
    if args.audio_unconditional is not None:
        cfg.train.audio_unconditional = bool(args.audio_unconditional)
    if args.objective is not None:
        cfg.cfm.objective = args.objective
    if args.t_sample_mode is not None:
        cfg.cfm.t_sample_mode = args.t_sample_mode
    if args.min_snr_gamma is not None:
        cfg.cfm.min_snr_gamma = args.min_snr_gamma
    if args.dropout is not None:
        cfg.model.dropout = args.dropout
    # Model size overrides. cond_dim defaults to dim if user passes --dim alone.
    if args.dim is not None:
        cfg.model.dim = args.dim
        if args.cond_dim is None:
            cfg.model.cond_dim = args.dim
    if args.n_layers is not None:
        cfg.model.n_layers = args.n_layers
    if args.n_heads is not None:
        cfg.model.n_heads = args.n_heads
    if args.head_dim is not None:
        cfg.model.head_dim = args.head_dim
    if args.cond_dim is not None:
        cfg.model.cond_dim = args.cond_dim
    if args.mlp_mult is not None:
        cfg.model.mlp_mult = args.mlp_mult

    train(cfg)


if __name__ == "__main__":
    main()
