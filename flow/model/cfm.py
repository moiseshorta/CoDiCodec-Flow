"""Conditional Flow Matching: training loss + ODE samplers.

We use the **rectified-flow / linear** CFM path:

    x_t = (1 - t) * x_data + t * eps,    eps ~ N(0, I),  t ~ p(t)

with target velocity

    v_target = dx/dt = eps - x_data

Time `t` is sampled per-sample from a logit-normal distribution
(SD3 schedule, Esser et al. 2024) which empirically gives faster convergence
than uniform sampling, with an option for plain uniform `t` for small
datasets where coverage of the tails matters more than concentration.

For continuation training we use a **per-token noise level**: prefix tokens
sit at `t=0` (clean) while target tokens carry a noise level `t ∈ (0, 1]`. The
loss is computed only on target tokens.

Latent normalization is handled by `model.normalize` / `model.denormalize`:
`cfm_loss` operates entirely in the model's normalized space (so the noise
`eps ~ N(0, I)` matches the boundary condition at `t=1`), and `sample` returns
latents in the original codec space ready to be fed to the decoder.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..utils import block_causal_mask


# --------------------------------------------------------------------------- #
# Time / prefix samplers
# --------------------------------------------------------------------------- #

def sample_logit_normal(batch: int, loc: float, scale: float, device: torch.device) -> torch.Tensor:
    """Sample t ~ sigmoid(N(loc, scale))."""
    z = torch.randn(batch, device=device) * scale + loc
    return torch.sigmoid(z)


def sample_uniform_t(batch: int, device: torch.device, t_min: float = 1e-3, t_max: float = 1.0) -> torch.Tensor:
    """Sample t ~ Uniform[t_min, t_max]. Useful for tiny datasets where
    logit-normal under-samples the [0, 0.1] tail (where the network has to
    learn the hardest "final denoising" steps).
    """
    return torch.rand(batch, device=device) * (t_max - t_min) + t_min


def sample_t(
    batch: int,
    mode: str,
    device: torch.device,
    logit_normal_loc: float = 0.0,
    logit_normal_scale: float = 1.0,
) -> torch.Tensor:
    """Dispatch on `mode` ("logit_normal" | "uniform")."""
    if mode == "logit_normal":
        return sample_logit_normal(batch, logit_normal_loc, logit_normal_scale, device=device)
    if mode == "uniform":
        return sample_uniform_t(batch, device=device)
    raise ValueError(f"unknown t sampling mode: {mode!r}")


def sample_prefix_chunks(
    n_chunks: int,
    p_unconditional: float,
    prefix_min_frac: float,
    prefix_max_frac: float,
) -> int:
    """Pick how many chunks at the start of the sequence are clean prompt.

    Returns the number of *chunks* (callers multiply by block_size to get tokens).
    With probability `p_unconditional` returns 0 (no prompt -> full unconditional).
    Otherwise samples uniformly from `[max(1, n*min), max(min, n*max)]`.
    """
    if random.random() < p_unconditional:
        return 0
    min_chunks = max(1, int(n_chunks * prefix_min_frac))
    max_chunks = max(min_chunks, int(n_chunks * prefix_max_frac))
    return random.randint(min_chunks, max_chunks)


# --------------------------------------------------------------------------- #
# Training loss
# --------------------------------------------------------------------------- #

@dataclass
class CFMOutputs:
    loss: torch.Tensor          # scalar
    v_pred: torch.Tensor        # [B, L, D]
    v_target: torch.Tensor      # [B, L, D]
    target_mask: torch.Tensor   # [B, L] (1.0 = target token, 0.0 = prefix/ignored)
    t_per_token: torch.Tensor   # [B, L] (noise level used)


def cfm_loss(
    model,
    x_data: torch.Tensor,
    prefix_len: int,
    t_sample: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    min_snr_gamma: Optional[float] = None,
) -> CFMOutputs:
    """One CFM training step (loss only, no optim).

    Args:
        model: a `FlowDiT` (provides `.normalize` / `.denormalize` and
            `.block_size`).
        x_data: [B, L, D] clean target in **raw codec-latent space**. The
            model normalizes internally; this function takes raw latents to
            keep the dataloader simple.
        prefix_len: number of leading tokens to keep clean (per-batch scalar).
        t_sample: [B] sampled noise levels in [0, 1].
        attn_mask: optional precomputed [L, L] additive mask.
        min_snr_gamma: if set, applies Min-SNR-γ loss weighting (Hang et al.
            2023) per-batch sample. The weights are renormalized so the
            mean over a batch is 1, which keeps the magnitude comparable to
            the unweighted MSE for logging.
    """
    # Move into normalized space so v_target = eps - x is well-conditioned
    # and the boundary condition at t=1 is exactly N(0, I).
    x_data = model.normalize(x_data)

    B, L, D = x_data.shape
    device = x_data.device

    eps = torch.randn_like(x_data)

    # Per-token target mask: 1.0 for target positions, 0.0 for prefix.
    is_target = torch.arange(L, device=device) >= prefix_len  # [L]
    target_mask = is_target.float().unsqueeze(0).expand(B, -1).contiguous()  # [B, L]

    # Per-token noise level: prefix at 0, target at t_sample.
    t_per_token = target_mask * t_sample.view(B, 1)  # [B, L]

    # Build noisy x_t: prefix clean, target = (1-t)*x + t*eps.
    t_b = t_per_token.unsqueeze(-1)  # [B, L, 1]
    x_t = (1.0 - t_b) * x_data + t_b * eps
    # (Prefix gets t=0 -> x_t = x_data, so no special handling needed.)

    v_target = eps - x_data  # rectified flow velocity

    if attn_mask is None:
        attn_mask = block_causal_mask(L, model.block_size, device=device).to(x_data.dtype)

    v_pred = model(x_t, t_per_token, attn_mask=attn_mask)

    # MSE per token, masked to target positions only.
    sq = (v_pred - v_target).pow(2).mean(dim=-1)  # [B, L]
    weighted_mask = target_mask

    if min_snr_gamma is not None and min_snr_gamma > 0:
        # For rectified flow: alpha=1-t, sigma=t -> SNR = (1-t)^2 / t^2.
        # For v-prediction the optimal MSE weighting is
        #   w(t) = min(SNR, gamma) / (SNR + 1)
        # (Hang et al. 2023, eq. 7 specialized to v-pred). We renormalize so
        # the expected weight over the batch is 1, otherwise the reported
        # loss number changes meaning.
        t_for_w = t_sample.clamp(min=1e-3, max=1.0 - 1e-3)
        snr = (1.0 - t_for_w).pow(2) / t_for_w.pow(2)
        w = torch.minimum(snr, snr.new_full(snr.shape, float(min_snr_gamma))) / (snr + 1.0)
        w = w / w.mean().clamp(min=1e-6)
        weighted_mask = target_mask * w.view(B, 1)

    denom = weighted_mask.sum().clamp(min=1.0)
    loss = (sq * weighted_mask).sum() / denom

    return CFMOutputs(
        loss=loss,
        v_pred=v_pred,
        v_target=v_target,
        target_mask=target_mask,
        t_per_token=t_per_token,
    )


# --------------------------------------------------------------------------- #
# ODE samplers (inference)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sample(
    model,
    prefix: torch.Tensor,
    n_target_tokens: int,
    n_steps: int,
    solver: str = "heun",
    eta: float = 0.0,
) -> torch.Tensor:
    """Sample `n_target_tokens` continuation tokens given a clean `prefix`.

    Args:
        model: trained FlowDiT.
        prefix: [B, P, D] clean prompt latents in **raw codec space**
                (P may be 0 for unconditional generation).
        n_target_tokens: number of tokens to generate (must be a multiple of
            `model.block_size`).
        n_steps: number of ODE steps.
        solver: 'euler' | 'heun'.
        eta: stochasticity level (0 = deterministic). Currently unused; kept as
            a hook for future SDE-based samplers.

    Returns:
        Tensor [B, P + n_target_tokens, D] in **raw codec space**, ready to
        feed into `CodecWrapper.decode_latents`.
    """
    assert n_target_tokens % model.block_size == 0, (
        f"n_target_tokens={n_target_tokens} must be a multiple of block_size={model.block_size}"
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if prefix is None:
        B = 1
        prefix = torch.zeros(B, 0, model.latent_dim, device=device, dtype=dtype)
    elif prefix.dim() == 3 and prefix.shape[1] == 0:
        # Empty prefix with explicit batch dim -> unconditional, B from caller.
        B = prefix.shape[0]
        prefix = prefix.to(device=device, dtype=dtype)
    else:
        prefix = prefix.to(device=device, dtype=dtype)
        B = prefix.shape[0]

    P = prefix.shape[1]
    L = P + n_target_tokens
    if L > model.max_seq_len:
        raise ValueError(f"L={L} > max_seq_len={model.max_seq_len}")

    # Normalize the prefix so the model sees a unit-Gaussian-distributed
    # observation for the clean tokens. The target noise is N(0, I) which
    # matches the boundary t=1 of the *normalized* path.
    if P > 0:
        prefix_norm = model.normalize(prefix)
    else:
        prefix_norm = prefix

    target_noise = torch.randn(B, n_target_tokens, model.latent_dim, device=device, dtype=dtype)
    x = torch.cat([prefix_norm, target_noise], dim=1)  # [B, L, D] in normalized space

    attn_mask = block_causal_mask(L, model.block_size, device=device).to(dtype)

    def t_per_token(t_now: float) -> torch.Tensor:
        is_target = torch.arange(L, device=device) >= P
        return is_target.float().unsqueeze(0).expand(B, -1) * t_now

    dt = 1.0 / n_steps
    for k in range(n_steps):
        t_now = 1.0 - k * dt
        t_next = max(0.0, t_now - dt)

        if solver == "euler":
            v = model(x, t_per_token(t_now), attn_mask=attn_mask)
            x = x.clone()
            x[:, P:] = x[:, P:] - dt * v[:, P:]
        elif solver == "heun":
            v1 = model(x, t_per_token(t_now), attn_mask=attn_mask)
            x_pred = x.clone()
            x_pred[:, P:] = x_pred[:, P:] - dt * v1[:, P:]
            v2 = model(x_pred, t_per_token(t_next), attn_mask=attn_mask)
            x = x.clone()
            x[:, P:] = x[:, P:] - 0.5 * dt * (v1[:, P:] + v2[:, P:])
        else:
            raise ValueError(f"unknown solver: {solver}")

    # Denormalize so the output sits in the codec's expected space.
    return model.denormalize(x)
