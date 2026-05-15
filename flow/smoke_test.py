"""Tiny smoke test that verifies the model + CFM loss + sampler are wired
correctly on the user's hardware. Intentionally does NOT touch the codec by
default (codec checkpoint must be downloaded ~600 MB on first use).

Run:
    python -m flow.smoke_test                # model-only, fast
    python -m flow.smoke_test --with-codec   # also encode/decode a sine
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .config import Config
from .model.cfm import cfm_loss, sample, sample_logit_normal, sample_prefix_chunks
from .model.dit import FlowDiT
from .train import build_model
from .utils import block_causal_mask, best_device, count_params, get_logger, human_int, set_seed

logger = get_logger("flow.smoke")


def smoke_model(device: torch.device) -> None:
    cfg = Config()
    # Shrink everything for quick test.
    cfg.model.dim = 128
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.head_dim = 32
    cfg.model.cond_dim = 128
    cfg.data.crop_tokens = 64  # 8 chunks
    cfg.train.batch_size = 2

    model = build_model(cfg.model).to(device)
    n = count_params(model)
    logger.info("smoke model has %s params (%s)", n, human_int(n))

    L = cfg.data.crop_tokens
    block_size = cfg.model.block_size
    B = cfg.train.batch_size
    D = cfg.model.latent_dim

    # Random latent batch (in unit-Gaussian space, like the codec output).
    x = torch.randn(B, L, D, device=device)
    t = sample_logit_normal(B, 0.0, 1.0, device=device)
    prefix_chunks = sample_prefix_chunks(L // block_size, 0.0, 0.25, 0.5)
    prefix_len = prefix_chunks * block_size
    logger.info("prefix_len=%d / %d tokens (%d chunks)", prefix_len, L, prefix_chunks)

    mask = block_causal_mask(L, block_size, device=device).to(x.dtype)

    t0 = time.time()
    out = cfm_loss(model, x, prefix_len=prefix_len, t_sample=t, attn_mask=mask)
    loss_val = out.loss.detach().item()
    out.loss.backward()
    dt_train = time.time() - t0
    logger.info("forward+backward OK in %.3fs, loss=%.4f", dt_train, loss_val)

    # Sampling
    model.eval()
    prefix = torch.randn(1, 16, D, device=device)
    t0 = time.time()
    full = sample(model, prefix=prefix, n_target_tokens=24, n_steps=4, solver="heun")
    dt_samp = time.time() - t0
    logger.info("sampled (Heun, 4 NFE) shape=%s in %.3fs", tuple(full.shape), dt_samp)


def smoke_codec(device: torch.device) -> None:
    """Round-trip a 2-second sine through the codec. Slow on first run."""
    from .codec_wrapper import CodecConfig, CodecWrapper
    codec = CodecWrapper(CodecConfig(device=str(device)))
    sr = codec.sample_rate
    n = sr * 2
    t = np.linspace(0, 2, n, endpoint=False, dtype=np.float32)
    sine = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)], axis=0).astype(np.float32)
    wv = torch.from_numpy(sine)
    logger.info("encoding 2s sine ...")
    t0 = time.time()
    lat = codec.encode_audio(wv, sr=sr)
    logger.info("encoded in %.2fs, latent shape=%s", time.time() - t0, tuple(lat.shape))
    t0 = time.time()
    rec = codec.decode_latents(lat)
    logger.info("decoded in %.2fs, audio shape=%s", time.time() - t0, tuple(rec.shape))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--with-codec", action="store_true")
    args = p.parse_args()

    set_seed(0)
    device = best_device(args.device)
    logger.info("device=%s", device)

    smoke_model(device)
    if args.with_codec:
        smoke_codec(device)
    logger.info("smoke test passed.")


if __name__ == "__main__":
    main()
