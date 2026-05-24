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

import math
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
# ODE / SDE samplers (inference)
# --------------------------------------------------------------------------- #
#
# Convention: t=1 is pure noise N(0,I), t=0 is clean data, and the model
# predicts the rectified-flow velocity v = eps - x_data. Integration goes
# backwards from t=1 to t=0 with dt = t_next - t_curr (< 0):
#
#     x(t_next) = x(t_curr) + dt * v(x, t_curr)              (Euler)
#
# Higher-order solvers (Heun / midpoint / RK4) follow the standard explicit
# Runge-Kutta updates. DPM-Solver++ uses the closed-form rectified-flow
# update with optional second-order Richardson extrapolation. Ping-pong is
# a stochastic SDE-style sampler best suited to distilled models.

SUPPORTED_SOLVERS = ("euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong")
SUPPORTED_SCHEDULES = ("linear", "shifted")


def _build_schedule(
    n_steps: int,
    schedule: str,
    schedule_shift: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a 1D tensor of shape (n_steps + 1,) with t in [0, 1] from 1 -> 0.

    Args:
        schedule: 'linear' (uniform) or 'shifted' (sigmoid logSNR shift -- a
            positive `schedule_shift` allocates more steps near t=1, useful
            for long sequences; a negative value crowds steps near t=0).
        schedule_shift: shift exponent in log-SNR space. Equivalent to the
            time-shifting trick used in SD3 / Stable Audio 3.
    """
    t = torch.linspace(1.0, 0.0, n_steps + 1, device=device, dtype=torch.float32)
    if schedule == "linear":
        pass
    elif schedule == "shifted":
        if schedule_shift != 0.0:
            # Apply a logSNR shift: t' = mu*t / (1 + (mu - 1) * t) with mu = exp(shift).
            # mu > 1 pushes mass toward t=1 (more denoising near pure noise).
            mu = float(math.exp(schedule_shift))
            # Avoid division issues at exact 0/1.
            t_clamped = t.clamp(min=1e-6, max=1.0 - 1e-6)
            t_shifted = mu * t_clamped / (1.0 + (mu - 1.0) * t_clamped)
            t = torch.where(t > 0.999999, t.new_tensor(1.0), t_shifted)
            t = torch.where(t < 1e-6, t.new_tensor(0.0), t)
    else:
        raise ValueError(f"unknown schedule: {schedule!r} (expected one of {SUPPORTED_SCHEDULES})")
    return t.to(dtype)


@torch.no_grad()
def sample(
    model,
    prefix: torch.Tensor,
    n_target_tokens: int,
    n_steps: int,
    solver: str = "heun",
    eta: float = 0.0,
    schedule: str = "linear",
    schedule_shift: float = 0.0,
) -> torch.Tensor:
    """Sample `n_target_tokens` continuation tokens given a clean `prefix`.

    Args:
        model: trained FlowDiT.
        prefix: [B, P, D] clean prompt latents in **raw codec space**
                (P may be 0 for unconditional generation).
        n_target_tokens: number of tokens to generate (must be a multiple of
            `model.block_size`).
        n_steps: number of ODE/SDE steps (NFE counted as model evaluations
            depends on the solver: euler=1, heun/midpoint=2, rk4=4, dpmpp=1,
            pingpong=1 per step).
        solver: one of `SUPPORTED_SOLVERS`:
            - 'euler'    : 1st-order ODE (1 NFE/step). Fastest baseline.
            - 'heun'     : 2nd-order ODE (2 NFE/step). Default; good quality.
            - 'midpoint' : 2nd-order RK2 (2 NFE/step). Often slightly cleaner
              than Heun on RF.
            - 'rk4'      : 4th-order Runge-Kutta (4 NFE/step). Highest quality
              for low step counts.
            - 'dpmpp'    : DPM-Solver++ 2M for rectified flow (1 NFE/step,
              uses 2nd-order Richardson extrapolation across consecutive
              steps). Strong quality at very low NFE (4-8).
            - 'pingpong' : stochastic SDE sampler (1 NFE/step). Recommended
              only for distilled / few-step models.
        eta: stochasticity level (currently only consumed by 'pingpong'
            implicitly through re-noising; kept for API compatibility).
        schedule: 'linear' or 'shifted' time grid (see `_build_schedule`).
        schedule_shift: logSNR shift applied when `schedule='shifted'`.

    Returns:
        Tensor [B, P + n_target_tokens, D] in **raw codec space**, ready to
        feed into `CodecWrapper.decode_latents`.
    """
    if solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"unknown solver: {solver!r} (expected one of {SUPPORTED_SOLVERS})")
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

    is_target = torch.arange(L, device=device) >= P  # [L]

    def t_per_token(t_now: float) -> torch.Tensor:
        return is_target.float().unsqueeze(0).expand(B, -1) * t_now

    def velocity(state: torch.Tensor, t_now: float) -> torch.Tensor:
        """Evaluate the model and return v restricted to target tokens.

        The prefix tokens are kept clean (t=0) so the model sees a fully
        consistent block-causal context.
        """
        return model(state, t_per_token(t_now), attn_mask=attn_mask)

    schedule_t = _build_schedule(n_steps, schedule, schedule_shift, device=device, dtype=torch.float32)

    if solver == "dpmpp":
        x = _sample_dpmpp(x, P, schedule_t, velocity)
    elif solver == "pingpong":
        x = _sample_pingpong(x, P, schedule_t, velocity)
    else:
        x = _sample_rk(x, P, schedule_t, velocity, solver=solver)

    # Denormalize so the output sits in the codec's expected space.
    return model.denormalize(x)


def _apply_target_update(x: torch.Tensor, P: int, delta: torch.Tensor) -> torch.Tensor:
    """Return a new tensor where target tokens (>= P) are replaced by delta."""
    out = x.clone()
    out[:, P:] = delta
    return out


def _sample_rk(
    x: torch.Tensor,
    P: int,
    schedule_t: torch.Tensor,
    velocity,
    solver: str,
) -> torch.Tensor:
    """Explicit Runge-Kutta family: euler, heun, midpoint, rk4.

    Updates only target tokens (positions >= P). Prefix tokens are kept clean
    via the `velocity` closure (which sets t=0 for the prefix).
    """
    n_steps = schedule_t.shape[0] - 1
    for i in range(n_steps):
        t_curr = float(schedule_t[i].item())
        t_next = float(schedule_t[i + 1].item())
        dt = t_next - t_curr  # negative

        if solver == "euler":
            k1 = velocity(x, t_curr)
            x = _apply_target_update(x, P, x[:, P:] + dt * k1[:, P:])
        elif solver == "heun":
            k1 = velocity(x, t_curr)
            x_pred = _apply_target_update(x, P, x[:, P:] + dt * k1[:, P:])
            k2 = velocity(x_pred, t_next)
            x = _apply_target_update(x, P, x[:, P:] + 0.5 * dt * (k1[:, P:] + k2[:, P:]))
        elif solver == "midpoint":
            t_mid = t_curr + 0.5 * dt
            k1 = velocity(x, t_curr)
            x_mid = _apply_target_update(x, P, x[:, P:] + 0.5 * dt * k1[:, P:])
            k2 = velocity(x_mid, t_mid)
            x = _apply_target_update(x, P, x[:, P:] + dt * k2[:, P:])
        elif solver == "rk4":
            t_mid = t_curr + 0.5 * dt
            k1 = velocity(x, t_curr)
            x2 = _apply_target_update(x, P, x[:, P:] + 0.5 * dt * k1[:, P:])
            k2 = velocity(x2, t_mid)
            x3 = _apply_target_update(x, P, x[:, P:] + 0.5 * dt * k2[:, P:])
            k3 = velocity(x3, t_mid)
            x4 = _apply_target_update(x, P, x[:, P:] + dt * k3[:, P:])
            # Avoid evaluating the model at exactly t=0 (not seen in training).
            t_eval = max(t_next, 1e-5)
            k4 = velocity(x4, t_eval)
            update = (dt / 6.0) * (k1[:, P:] + 2.0 * k2[:, P:] + 2.0 * k3[:, P:] + k4[:, P:])
            x = _apply_target_update(x, P, x[:, P:] + update)
        else:
            raise ValueError(f"unknown RK solver: {solver}")
    return x


def _sample_dpmpp(
    x: torch.Tensor,
    P: int,
    schedule_t: torch.Tensor,
    velocity,
) -> torch.Tensor:
    """DPM-Solver++ 2M for rectified flow.

    Closed-form rectified-flow update with second-order Richardson
    extrapolation across consecutive steps. Mirrors the implementation in
    Stability-AI/stable-audio-3 (`sample_flow_dpmpp`).
    """
    eps = 1e-10
    n_steps = schedule_t.shape[0] - 1
    old_denoised = None
    log_snr = lambda t: math.log(max(1.0 - t, eps) / max(t, eps))

    for i in range(n_steps):
        t_curr = float(schedule_t[i].item())
        t_next = float(schedule_t[i + 1].item())
        t_prev = float(schedule_t[i - 1].item()) if i > 0 else None

        v = velocity(x, t_curr)
        # x_data prediction in normalized space: denoised = x - t * v.
        denoised = x[:, P:] - t_curr * v[:, P:]

        alpha_t = 1.0 - t_next
        # Closed-form coefficient: dt / [(1 - t_next) * t_curr], matches
        # (-h).expm1() in log-SNR formulation but avoids numerical issues.
        denom = max(1.0 - t_next, eps) * max(t_curr, eps)
        dpmpp_coeff = (t_next - t_curr) / denom

        is_first_step = old_denoised is None
        is_last_step = (t_next == 0.0)

        if is_first_step or is_last_step:
            denoised_use = denoised
        else:
            # Second-order multistep correction in log-SNR space.
            h = log_snr(t_next) - log_snr(t_curr)
            h_last = log_snr(t_curr) - log_snr(t_prev)  # type: ignore[arg-type]
            r = h_last / h if h != 0.0 else 0.0
            if r == 0.0:
                denoised_use = denoised
            else:
                denoised_use = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised

        scale = t_next / max(t_curr, eps)
        target_new = scale * x[:, P:] - alpha_t * dpmpp_coeff * denoised_use
        x = _apply_target_update(x, P, target_new)
        old_denoised = denoised
    return x


def _sample_pingpong(
    x: torch.Tensor,
    P: int,
    schedule_t: torch.Tensor,
    velocity,
) -> torch.Tensor:
    """Ping-pong (SDE) sampler for rectified flow.

    Each step denoises to x_data (using the current velocity) and then
    re-noises with fresh Gaussian noise to the next noise level. Best suited
    to distilled / few-step models (Stable Audio 3 uses this for its
    consistency-distilled checkpoint).
    """
    n_steps = schedule_t.shape[0] - 1
    for i in range(n_steps):
        t_curr = float(schedule_t[i].item())
        t_next = float(schedule_t[i + 1].item())

        v = velocity(x, t_curr)
        denoised = x[:, P:] - t_curr * v[:, P:]
        if t_next > 0.0:
            new_noise = torch.randn_like(denoised)
            target_new = (1.0 - t_next) * denoised + t_next * new_noise
        else:
            target_new = denoised
        x = _apply_target_update(x, P, target_new)
    return x
