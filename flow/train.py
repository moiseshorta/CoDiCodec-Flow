"""Training loop for the block-causal Flow-Matching DiT.

This script is intentionally dependency-light (no hydra, no wandb): plain
argparse + dataclasses + tqdm. Designed to be runnable on a single GPU (CUDA
or MPS) with 36 GB of unified memory.

Usage:
    python -m flow.train \
        --data-dir ./data/latents \
        --out-dir  ./runs/v0      \
        --device   mps            \
        --batch-size 8 --max-steps 200000

Checkpoints are written every `--ckpt-every` steps:
    {out_dir}/last.pt   - model + optim + scheduler + EMA, for resuming
    {out_dir}/ema.pt    - EMA-only state_dict, for inference
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import (
    CODEC_CHUNK_SAMPLES,
    CFMConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
)
from .data.lat_stats import LatStats, load_or_compute as load_or_compute_lat_stats
from .data.latent_dataset import LatentDataset
from .model.cfm import cfm_loss, sample as cfm_sample, sample_prefix_chunks, sample_t
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
)


logger = get_logger("flow.train")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def build_model(cfg: ModelConfig, lat_stats: Optional[LatStats] = None) -> FlowDiT:
    """Construct a FlowDiT, optionally baking in dataset latent normalization."""
    return FlowDiT(
        latent_dim=cfg.latent_dim,
        block_size=cfg.block_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        mlp_mult=cfg.mlp_mult,
        cond_dim=cfg.cond_dim,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout,
        lat_mean=(lat_stats.mean if lat_stats is not None else None),
        lat_std=(lat_stats.std if lat_stats is not None else None),
    )


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
    logger.info("device=%s dtype=%s", device, dtype)

    out_dir = ensure_dir(cfg.train.out_dir)

    # ------- data -------------------------------------------------------- #
    train_ds = LatentDataset(
        cfg.data.data_dir,
        crop_tokens=cfg.data.crop_tokens,
        seed=cfg.data.seed,
        split="train",
        val_frac=cfg.data.val_frac,
    )
    val_ds = LatentDataset(
        cfg.data.data_dir,
        crop_tokens=cfg.data.crop_tokens,
        seed=cfg.data.seed,
        split="val",
        val_frac=cfg.data.val_frac,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.data.num_workers > 0,
    )
    # Note: validation iterates `val_ds` directly via `evaluate()`, which
    # builds a deterministic (crop, t, prefix) grid -- no DataLoader needed.

    # ------- latent normalization stats ---------------------------------- #
    # Stats are computed once (cached as `lat_stats.pt` under data_dir) over
    # all train shards; the resulting per-channel mean/std are baked into the
    # model as buffers so checkpoints are self-contained.
    train_shard_paths = [s.path for s in train_ds.shards]
    lat_stats = load_or_compute_lat_stats(cfg.data.data_dir, shard_paths=train_shard_paths)

    # ------- model ------------------------------------------------------- #
    model = build_model(cfg.model, lat_stats=lat_stats).to(device)
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
    )

    # GradScaler is only useful with fp16 on CUDA; bf16 doesn't need it.
    scaler = torch.amp.GradScaler(device.type) if (dtype == torch.float16 and device.type == "cuda") else None

    # Pre-build attn mask for the default crop length.
    L = cfg.data.crop_tokens
    block_size = cfg.model.block_size
    attn_mask_const = block_causal_mask(L, block_size, device=device)

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
        sd = torch.load(str(last_ckpt), map_location="cpu", weights_only=False)
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
        sd = torch.load(str(init_path), map_location="cpu", weights_only=False)
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
                ema.buffers[name] = b.detach().clone().cpu()
            logger.info("init-from: seeded EMA shadow from source checkpoint")

    # ------- train loop -------------------------------------------------- #
    model.train()
    train_iter = iter(train_dl)
    t0 = time.time()
    accum = 0
    running = {"loss": 0.0, "n": 0}
    # Lazy CodecWrapper holder; only built when audio sampling fires.
    codec_state: dict = {"codec": None}

    # Generate samples at step 0 so we hear what untrained model sounds like.
    if cfg.train.audio_sample_every > 0:
        try:
            generate_audio_samples(
                model, ema, val_ds, cfg, device,
                step=start_step, out_dir=out_dir, codec_state=codec_state,
            )
        except Exception as e:
            logger.warning("initial audio sample failed (continuing): %s", e)

    for step in range(start_step, cfg.train.max_steps):
        # Cosine LR schedule.
        lr = cosine_lr(step, cfg.train.warmup_steps, cfg.train.max_steps, cfg.train.lr)
        for g in optim.param_groups:
            g["lr"] = lr

        # Grab a batch (loop forever).
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        x_data = batch.to(device, non_blocking=True)  # [B, L, D]
        B = x_data.shape[0]

        # Sample t and prefix length.
        t_sample = sample_t(
            B,
            mode=cfg.cfm.t_sample_mode,
            device=device,
            logit_normal_loc=cfg.cfm.logit_normal_loc,
            logit_normal_scale=cfg.cfm.logit_normal_scale,
        )
        n_chunks = L // block_size
        prefix_chunks = sample_prefix_chunks(
            n_chunks,
            cfg.cfm.p_unconditional,
            cfg.cfm.prefix_min_frac,
            cfg.cfm.prefix_max_frac,
        )
        prefix_len = prefix_chunks * block_size

        # Forward + loss in autocast.
        min_snr = cfg.cfm.min_snr_gamma if cfg.cfm.min_snr_gamma and cfg.cfm.min_snr_gamma > 0 else None
        with autocast_ctx(device, dtype=dtype, enabled=(dtype != torch.float32)):
            out = cfm_loss(
                model,
                x_data=x_data,
                prefix_len=prefix_len,
                t_sample=t_sample,
                attn_mask=attn_mask_const,
                min_snr_gamma=min_snr,
            )
            loss = out.loss / cfg.train.grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum += 1
        running["loss"] += float(out.loss.detach().cpu()) * B
        running["n"] += B

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
        if (step + 1) % cfg.train.log_every == 0 and running["n"] > 0:
            avg = running["loss"] / running["n"]
            elapsed = time.time() - t0
            steps_per_sec = (step + 1 - start_step) / max(1e-6, elapsed)
            logger.info(
                "step=%d/%d  loss=%.4f  lr=%.2e  t=%.3f  prefix=%d/%d  steps/s=%.2f",
                step + 1,
                cfg.train.max_steps,
                avg,
                lr,
                float(t_sample.mean().cpu()),
                prefix_len,
                L,
                steps_per_sec,
            )
            running = {"loss": 0.0, "n": 0}

        # Validation
        if (step + 1) % cfg.train.val_every == 0:
            val_loss = evaluate(model, val_ds, cfg, device, dtype, attn_mask_const)
            logger.info("[val] step=%d  loss=%.4f", step + 1, val_loss)
            model.train()

        # Audio sampling
        if cfg.train.audio_sample_every > 0 and (step + 1) % cfg.train.audio_sample_every == 0:
            try:
                generate_audio_samples(
                    model, ema, val_ds, cfg, device,
                    step=step + 1, out_dir=out_dir, codec_state=codec_state,
                )
            except Exception as e:
                logger.warning("audio sample at step %d failed (continuing): %s", step + 1, e)

        # Checkpoint
        if (step + 1) % cfg.train.ckpt_every == 0 or (step + 1) == cfg.train.max_steps:
            save_checkpoint(out_dir, model, optim, ema, cfg, step + 1)

    # Final save
    save_checkpoint(out_dir, model, optim, ema, cfg, cfg.train.max_steps)
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
    """Sample a few audio continuations from val prompts and save as .wav.

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
        codec_state["sample_rate"] = codec_state["codec"].sample_rate
    codec = codec_state["codec"]
    sr = codec_state["sample_rate"]

    samples_dir = ensure_dir(os.path.join(out_dir, "samples"))

    # Optional: swap in EMA weights for sampling.
    backup: dict[str, torch.Tensor] = {}
    if cfg.train.audio_use_ema:
        backup = {n: p.data.clone() for n, p in model.named_parameters()}
        ema.copy_to(model)

    was_training = model.training
    model.eval()

    block_size = cfg.model.block_size
    samples_per_chunk = codec.samples_per_chunk
    unconditional = bool(cfg.train.audio_unconditional)

    # Total generated length (in chunks) is the same in both modes:
    # prompt_seconds + continuation_seconds. In unconditional mode the model
    # generates the entire span from noise; in continuation mode the first
    # `prompt_seconds` is taken from a val clip and held clean.
    prompt_chunks = max(1, int(round(cfg.train.audio_prompt_seconds * sr / samples_per_chunk)))
    cont_chunks = max(1, int(round(cfg.train.audio_continuation_seconds * sr / samples_per_chunk)))

    if unconditional:
        # All tokens are generated; no prompt.
        prompt_tokens = 0
        cont_tokens = (prompt_chunks + cont_chunks) * block_size
    else:
        prompt_tokens = prompt_chunks * block_size
        cont_tokens = cont_chunks * block_size

    if prompt_tokens + cont_tokens > cfg.model.max_seq_len:
        logger.warning(
            "audio sample length (%d) exceeds max_seq_len (%d); truncating",
            prompt_tokens + cont_tokens, cfg.model.max_seq_len,
        )
        cont_tokens = (cfg.model.max_seq_len - prompt_tokens) // block_size * block_size

    n = max(1, cfg.train.audio_n_samples)
    if not unconditional:
        n = min(n, len(val_ds))
        indices = torch.randperm(len(val_ds))[:n].tolist()
    else:
        indices = list(range(n))  # synthetic indices for filename only

    try:
        for i, idx in enumerate(indices):
            if unconditional:
                # Empty prefix with explicit batch dim -> cfm_sample uses B=1.
                prompt = torch.zeros(1, 0, cfg.model.latent_dim, device=device)
            else:
                crop = val_ds[idx]  # [crop_tokens, 64], CPU
                if crop.shape[0] < prompt_tokens:
                    continue
                prompt = crop[:prompt_tokens].unsqueeze(0).to(device)  # [1, P, 64]

            try:
                full = cfm_sample(
                    model,
                    prefix=prompt,
                    n_target_tokens=cont_tokens,
                    n_steps=cfg.train.audio_nfe,
                    solver=cfg.train.audio_solver,
                )  # [1, P+T, 64]
            except Exception as e:
                logger.warning("audio sample %d: sampling failed: %s", i, e)
                continue

            full = full.squeeze(0).to(torch.float32)
            try:
                wv = codec.decode_latents(full)  # [2, N] expected
            except Exception as e:
                logger.warning("audio sample %d: decode failed: %s", i, e)
                continue

            if wv.dim() == 3:
                wv = wv[0]
            wv_np = wv.transpose(0, 1).contiguous().cpu().numpy().astype(np.float32)
            # Soft-clip very loud excursions so users don't get blown out by garbage at init.
            wv_np = np.clip(wv_np, -1.0, 1.0)
            out_path = os.path.join(samples_dir, f"step_{step:07d}_idx_{i:02d}.wav")
            sf.write(out_path, wv_np, sr)
            mode = "uncond" if unconditional else "cont"
            logger.info(
                "[audio] step=%d idx=%d (%s) -> %s (%.2fs)",
                step, i, mode, out_path,
                (prompt_tokens + cont_tokens) * samples_per_chunk / block_size / sr,
            )
    finally:
        # Restore non-EMA weights and training mode.
        if backup:
            for nname, p in model.named_parameters():
                if nname in backup:
                    p.data.copy_(backup[nname])
        if was_training:
            model.train()


@torch.no_grad()
def evaluate(
    model,
    val_ds: LatentDataset,
    cfg: Config,
    device,
    dtype,
    attn_mask,
    n_crops: int = 8,
    seed: int = 0,
) -> float:
    """Deterministic, low-variance validation loss.

    Averages CFM loss over a fixed grid of `(crop, t, prefix_chunks)`
    configurations, with seeded noise. This makes val loss reproducible
    across calls (so trends are meaningful instead of random fluctuation),
    and gives a stable estimate by integrating over the (t, prefix) surface
    that training also samples from.

    Total configs evaluated per call:
        n_shards * n_crops * |T_GRID| * |PREFIX_GRID|

    For our small val set (1 shard) with the defaults below this is
    1 * 8 * 5 * 4 = 160 configurations, batched into 8 forward passes of
    batch size 20 each.
    """
    model.eval()
    L = cfg.data.crop_tokens
    block_size = cfg.model.block_size
    n_chunks = L // block_size

    # Cover the t-axis the model is trained on (logit-normal sits around 0.5
    # but we want the tails too).
    t_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Cover the prefix-fraction axis: 0% (unconditional), 25%, 50%, 75%.
    prefix_grid = [
        0,
        max(1, n_chunks // 4),
        max(1, n_chunks // 2),
        max(1, (3 * n_chunks) // 4),
    ]
    grid = [(t, p) for t in t_grid for p in prefix_grid]
    G = len(grid)

    # Pre-build per-config target masks and t-per-token tensors (constant
    # across crops, computed once on device).
    arange_L = torch.arange(L, device=device)
    target_masks = torch.zeros(G, L, device=device, dtype=torch.float32)
    t_per_token = torch.zeros(G, L, device=device, dtype=torch.float32)
    for g_idx, (t_val, prefix_chunks) in enumerate(grid):
        prefix_len = prefix_chunks * block_size
        is_target = (arange_L >= prefix_len).float()
        target_masks[g_idx] = is_target
        t_per_token[g_idx] = is_target * float(t_val)
    denoms = target_masks.sum(dim=-1).clamp(min=1.0)  # [G]

    # Seeded RNG so val loss is reproducible across training steps.
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    total_loss = 0.0
    n_total = 0

    for meta in val_ds.shards:
        obj = torch.load(str(meta.path), map_location="cpu", weights_only=False)
        latent = obj["latent"]  # [T, 8, D]
        flat = latent.reshape(-1, latent.shape[-1]).float()  # [T*8, D]
        n_tokens = flat.shape[0]
        max_start_chunks = (n_tokens - L) // block_size
        if max_start_chunks < 0:
            continue
        # Evenly-spaced deterministic crops across the shard.
        if max_start_chunks == 0 or n_crops == 1:
            crop_starts = [0]
        else:
            crop_starts = torch.linspace(0, max_start_chunks, n_crops).long().tolist()

        for start_chunk in crop_starts:
            start = start_chunk * block_size
            crop_raw = flat[start : start + L].to(device)  # [L, D] in raw codec space
            # Move into model's normalized space *once* per crop. The model
            # itself expects normalized inputs (see FlowDiT.forward).
            crop = model.normalize(crop_raw)
            x_data = crop.unsqueeze(0).expand(G, -1, -1).contiguous()  # [G, L, D]

            # Seeded eps so two evaluations of the same checkpoint return
            # the same loss.
            eps = torch.randn(x_data.shape, generator=cpu_gen).to(device=device)

            t_b = t_per_token.unsqueeze(-1)  # [G, L, 1]
            x_t = (1.0 - t_b) * x_data + t_b * eps
            v_target = eps - x_data

            with autocast_ctx(device, dtype=dtype, enabled=(dtype != torch.float32)):
                v_pred = model(x_t, t_per_token, attn_mask=attn_mask)

            sq = (v_pred.float() - v_target).pow(2).mean(dim=-1)  # [G, L]
            losses_per_g = (sq * target_masks).sum(dim=-1) / denoms  # [G]
            total_loss += float(losses_per_g.sum().detach().cpu())
            n_total += G

    return total_loss / max(1, n_total)


def save_checkpoint(out_dir: str, model, optim, ema: EMA, cfg: Config, step: int) -> None:
    last = Path(out_dir) / "last.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "ema": ema.state_dict(),
            "config": {
                "model": asdict(cfg.model),
                "cfm": asdict(cfg.cfm),
                "data": asdict(cfg.data),
                "train": asdict(cfg.train),
                "device": cfg.device,
            },
        },
        str(last),
    )
    # Standalone EMA file for inference.
    ema_path = Path(out_dir) / "ema.pt"
    torch.save(
        {
            "ema": ema.state_dict(),
            "config": {
                "model": asdict(cfg.model),
                "cfm": asdict(cfg.cfm),
            },
            "step": step,
        },
        str(ema_path),
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
    p.add_argument("--device", default=None, help="cuda | mps | cpu")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dtype", default=None, help="fp32 | bf16 | fp16")
    p.add_argument("--crop-tokens", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    # Audio sampling
    p.add_argument("--audio-sample-every", type=int, default=None,
                   help="0 disables audio sampling during training.")
    p.add_argument("--audio-n-samples", type=int, default=None)
    p.add_argument("--audio-prompt-seconds", type=float, default=None)
    p.add_argument("--audio-continuation-seconds", type=float, default=None)
    p.add_argument("--audio-nfe", type=int, default=None)
    p.add_argument("--audio-solver", default=None, choices=["euler", "heun"])
    p.add_argument("--audio-unconditional", dest="audio_unconditional", action="store_true", default=None,
                   help="Generate audio samples unconditionally from pure noise (default).")
    p.add_argument("--audio-continuation", dest="audio_unconditional", action="store_false",
                   help="Generate audio samples by continuing a val-set prompt.")
    # CFM / loss knobs.
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
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.log_every:
        cfg.train.log_every = args.log_every
    if args.val_every:
        cfg.train.val_every = args.val_every
    if args.ckpt_every:
        cfg.train.ckpt_every = args.ckpt_every
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
