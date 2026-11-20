"""Tiny smoke test that verifies the model + losses + sampler are wired
correctly on the user's hardware. Intentionally does NOT touch the codec by
default (codec checkpoint must be downloaded ~600 MB on first use).

Run:
    python -m flow.smoke_test                # model-only, fast
    python -m flow.smoke_test --with-codec   # also checks segmented encoding + decode
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .config import Config
from .model.cfm import (
    SUPPORTED_SOLVERS,
    cfm_loss,
    generate,
    sample_joint,
    sample_logit_normal,
    sample_t,
    teacher_forcing_loss,
)
from .train import build_model
from .utils import block_causal_mask, best_device, count_params, get_logger, human_int, set_seed, teacher_forcing_mask

logger = get_logger("flow.smoke")


def _close(a: torch.Tensor, b: torch.Tensor, what: str, tol: float = 1e-4) -> None:
    err = ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()
    assert err < tol, f"{what}: relative error {err:.2e}"
    logger.info("%-45s OK (rel err %.1e)", what, err)


def smoke_model(device: torch.device) -> None:
    cfg = Config()
    # Shrink everything for quick test.
    cfg.model.dim = 128
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.model.head_dim = 32
    cfg.model.cond_dim = 128
    cfg.data.crop_tokens = 64  # 8 chunks

    model = build_model(cfg.model).to(device)
    n = count_params(model)
    logger.info("smoke model has %s params (%s)", n, human_int(n))
    # AdaLN-Zero starts every block as identity with zero output; randomize so
    # the equivalence checks below compare non-trivial functions.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)

    L, bs, B, D = cfg.data.crop_tokens, cfg.model.block_size, 2, cfg.model.latent_dim
    x = torch.randn(B, L, D, device=device)

    # Both training objectives run forward + backward.
    t0 = time.time()
    t_chunk = sample_t(B * L // bs, "uniform", device).view(B, -1)
    loss = teacher_forcing_loss(model, x, t_chunk, teacher_forcing_mask(L, bs, device))
    loss.backward()
    out = cfm_loss(model, x, prefix_len=2 * bs, t_sample=sample_logit_normal(B, 0.0, 1.0, device),
                   attn_mask=block_causal_mask(L, bs, device))
    out.loss.backward()
    logger.info("forward+backward OK in %.3fs (teacher_forcing=%.4f, joint=%.4f)",
                time.time() - t0, loss.item(), out.loss.item())

    model.eval()
    with torch.no_grad():
        # Cached context K/V give the same velocity as running the whole window.
        ctx, tgt = x[:1, :3 * bs], x[:1, 3 * bs:4 * bs]
        t_tgt = torch.full((1, bs), 0.7, device=device)
        full = model(torch.cat([ctx, tgt], 1), torch.cat([torch.zeros(1, 3 * bs, device=device), t_tgt], 1))
        _, kv = model(ctx, torch.zeros(1, 3 * bs, device=device), return_kv=True)
        _close(model(tgt, t_tgt, kv=kv), full[:, 3 * bs:], "cached context == full window")

        # Teacher forcing trains exactly the generation condition: noisy chunk i
        # equals chunk i denoised on its clean preceding chunks.
        i = 5
        t_tok = t_chunk[:1].repeat_interleave(bs, 1)
        v_tf = model(torch.cat([x[:1], x[1:]], 1), torch.cat([torch.zeros_like(t_tok), t_tok], 1),
                     attn_mask=teacher_forcing_mask(L, bs, device),
                     pos=torch.arange(L, device=device).repeat(2))[:, L:]
        v_gen = model(torch.cat([x[:1, :i * bs], x[1:, i * bs:(i + 1) * bs]], 1),
                      torch.cat([torch.zeros(1, i * bs, device=device), t_tok[:, i * bs:(i + 1) * bs]], 1))
        _close(v_tf[:, i * bs:(i + 1) * bs], v_gen[:, i * bs:], "teacher-forced chunk == generation condition")

    # Every solver runs chunk by chunk and stays finite.
    prefix = torch.randn(1, 2 * bs, D, device=device)
    for solver in SUPPORTED_SOLVERS:
        t0 = time.time()
        full = generate(model, prefix, n_chunks=3, context_chunks=2, n_steps=4, solver=solver)
        assert full.shape == (1, 5 * bs, D), f"{solver}: bad shape {full.shape}"
        assert torch.isfinite(full).all(), f"{solver}: non-finite outputs"
        logger.info("generated (%-9s 4 steps) shape=%s in %.3fs", solver + ",", tuple(full.shape), time.time() - t0)
    full = generate(model, prefix[:, :0], n_chunks=2, n_steps=4, solver="heun", schedule="shifted", schedule_shift=1.0)
    assert torch.isfinite(full).all()
    logger.info("generated (heun, shifted schedule, unconditional) shape=%s", tuple(full.shape))
    full = sample_joint(model, prefix, 3 * bs, n_steps=4, solver="heun")
    assert full.shape == (1, 5 * bs, D) and torch.isfinite(full).all()
    logger.info("joint sampling (heun) shape=%s", tuple(full.shape))


def smoke_codec(device: torch.device) -> None:
    """Segmented encoding == whole-file encoding; decode round trip. Slow on first run."""
    from .codec_wrapper import CodecConfig, CodecWrapper
    from .data.preencode import encode_wave
    codec = CodecWrapper(CodecConfig(device=str(device)))
    sr, spc = codec.sample_rate, codec.samples_per_chunk
    n = 5 * spc + 3000
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(0)
    sig = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]) * 0.5
    wv = (sig + 0.05 * rng.standard_normal(sig.shape)).astype(np.float32)
    t0 = time.time()
    whole = encode_wave(codec, wv)
    split = encode_wave(codec, wv, max_seconds=2 * spc / sr)  # 2-chunk segments
    logger.info("encoded in %.2fs, latent shape=%s", time.time() - t0, tuple(whole.shape))
    assert split.shape == whole.shape, (split.shape, whole.shape)
    # Compared in the codec's bounded (tanh) space: atanh amplifies rounding near +-1.
    _close(torch.tanh(split * 0.8), torch.tanh(whole * 0.8), "segmented encoding == whole-file", tol=2e-2)
    t0 = time.time()
    rec = codec.decode_latents(whole)
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
