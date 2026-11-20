"""Conditional Flow Matching: training losses + chunk-by-chunk ODE samplers.

We use the **rectified-flow / linear** CFM path:

    x_t = (1 - t) * x_data + t * eps,    eps ~ N(0, I),  t ~ p(t)

with target velocity

    v_target = dx/dt = eps - x_data

Two training objectives:

- `cfm_loss` (joint, default): a clean prefix followed by a target span
  denoised jointly at one shared noise level.
- `teacher_forcing_loss`: the model sees [clean crop ; noisy crop] and noisy
  chunk i attends to clean chunks < i and to itself, so every chunk is trained
  in exactly the condition realtime generates it in, each with its own noise
  level. Learns the realtime condition much faster but overfits small datasets
  sooner (see README).

Two samplers: `generate` works chunk by chunk with a sliding window of clean
context (as realtime does; any length), `sample_joint` denoises a whole
continuation at once (the joint objective's condition; up to max_seq_len).

Latent normalization is handled by `model.normalize` / `model.denormalize`:
losses operate in the model's normalized space (so the noise `eps ~ N(0, I)`
matches the boundary condition at `t=1`), and the samplers return latents in
the original codec space ready to be fed to the decoder.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import torch


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


def min_snr_weight(t: torch.Tensor, gamma: float) -> torch.Tensor:
    """Min-SNR-γ weight for v-prediction (Hang et al. 2023, eq. 7) on the
    rectified-flow path: alpha=1-t, sigma=t -> SNR = (1-t)^2 / t^2,
    w = min(SNR, γ) / (SNR + 1)."""
    t = t.clamp(min=1e-3, max=1.0 - 1e-3)
    snr = (1.0 - t).pow(2) / t.pow(2)
    return snr.clamp(max=float(gamma)) / (snr + 1.0)


# --------------------------------------------------------------------------- #
# Training losses
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
    """Joint objective: one CFM training step (loss only, no optim).

    Args:
        model: a `FlowDiT` (provides `.normalize` / `.denormalize` and
            `.block_size`).
        x_data: [B, L, D] clean target in **raw codec-latent space**. The
            model normalizes internally; this function takes raw latents to
            keep the dataloader simple.
        prefix_len: number of leading tokens to keep clean (per-batch scalar,
            a multiple of block_size).
        t_sample: [B] sampled noise levels in [0, 1].
        attn_mask: optional precomputed [L, L] additive mask.
        min_snr_gamma: if set, applies Min-SNR-γ loss weighting per sample.
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

    v_pred = model(x_t, t_per_token, attn_mask=attn_mask)

    # MSE per token, masked to target positions only.
    sq = (v_pred - v_target).pow(2).mean(dim=-1)  # [B, L]
    weighted_mask = target_mask

    if min_snr_gamma is not None and min_snr_gamma > 0:
        # Renormalized to mean 1 so the logged loss keeps its meaning.
        w = min_snr_weight(t_sample, min_snr_gamma)
        weighted_mask = target_mask * (w / w.mean().clamp(min=1e-6)).view(B, 1)

    denom = weighted_mask.sum().clamp(min=1.0)
    loss = (sq * weighted_mask).sum() / denom

    return CFMOutputs(
        loss=loss,
        v_pred=v_pred,
        v_target=v_target,
        target_mask=target_mask,
        t_per_token=t_per_token,
    )


def teacher_forcing_loss(
    model,
    x_data: torch.Tensor,
    t_chunk: torch.Tensor,
    attn_mask: torch.Tensor,
    min_snr_gamma: Optional[float] = None,
    eps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Block-diffusion loss in the exact generation condition (cf. Arriola et al.
    2025, "Block Diffusion").

    The model sees [clean crop ; noisy crop] (2L tokens, RoPE positions repeated)
    under `utils.teacher_forcing_mask`: noisy chunk i attends to clean chunks < i
    and to itself, i.e. clean context -> one noisy chunk, for every chunk.

    Args:
        x_data: [B, L, D] clean crop in raw codec space.
        t_chunk: [B, L // block_size] independent noise level per chunk.
        attn_mask: `teacher_forcing_mask(L, block_size)`, shape [2L, 2L].
        eps: optional fixed noise (deterministic validation).
    Returns:
        Scalar loss over the noisy half.
    """
    x = model.normalize(x_data)
    B, L, _ = x.shape
    t = t_chunk.repeat_interleave(model.block_size, dim=1)  # [B, L]
    if eps is None:
        eps = torch.randn_like(x)
    x_t = (1.0 - t.unsqueeze(-1)) * x + t.unsqueeze(-1) * eps
    pos = torch.arange(L, device=x.device).repeat(2)
    v_pred = model(
        torch.cat([x, x_t], dim=1),
        torch.cat([torch.zeros_like(t), t], dim=1),
        attn_mask=attn_mask,
        pos=pos,
    )[:, L:]
    sq = (v_pred.float() - (eps - x)).pow(2).mean(dim=-1)  # [B, L]
    if min_snr_gamma:
        w = min_snr_weight(t, min_snr_gamma)
        return (sq * w).sum() / w.sum().clamp(min=1e-6)
    return sq.mean()


# --------------------------------------------------------------------------- #
# ODE / SDE samplers (inference)
# --------------------------------------------------------------------------- #
#
# Convention: t=1 is pure noise N(0,I), t=0 is clean data, and the model
# predicts the rectified-flow velocity v = eps - x_data. Integration goes
# backwards from t=1 to t=0 with dt = t_next - t_curr (< 0):
#
#     x(t_next) = x(t_curr) + dt * v(x, t_curr)              (Euler)

SUPPORTED_SOLVERS = ("euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong")
SUPPORTED_SCHEDULES = ("linear", "shifted")


def build_schedule(n_steps: int, schedule: str = "linear", schedule_shift: float = 0.0) -> list:
    """(n_steps + 1) noise levels from 1.0 (noise) to 0.0 (data), as floats.

    'shifted' applies the SD3 / Stable Audio logSNR shift
    t' = mu*t / (1 + (mu - 1)*t), mu = exp(shift): a positive shift spends more
    steps near t=1, a negative one near t=0.
    """
    if schedule not in SUPPORTED_SCHEDULES:
        raise ValueError(f"unknown schedule: {schedule!r} (expected one of {SUPPORTED_SCHEDULES})")
    n = max(1, int(n_steps))
    ts = [1.0 - k / n for k in range(n + 1)]
    if schedule == "shifted" and schedule_shift != 0.0:
        mu = math.exp(schedule_shift)
        ts = [t if t in (0.0, 1.0) else mu * t / (1.0 + (mu - 1.0) * t) for t in ts]
    return ts


def _log_snr(t: float) -> float:
    return math.log(max(1.0 - t, 1e-10) / max(t, 1e-10))


@torch.no_grad()
def integrate(x: torch.Tensor, ts: list, velocity, solver: str = "heun", temperature: float = 1.0) -> torch.Tensor:
    """Solve from noise (ts[0] = 1) to data (ts[-1] = 0).

    `velocity(x, t)` returns the model velocity for state x at noise level t.
    `temperature` scales it (v / temperature): 1 is the plain ODE, <1 overshoots
    toward the data, >1 leaves residual noise. Solvers: euler (1 model call per
    step), heun / midpoint (2), rk4 (4), dpmpp (DPM-Solver++ 2M, 1), pingpong
    (stochastic, 1; best for distilled models).

    Heun and RK4 take a plain Euler step into t=0: t=0 is the clean-context
    level, which never receives a loss, so the velocity there is untrained.
    """
    if solver not in SUPPORTED_SOLVERS:
        raise ValueError(f"unknown solver: {solver!r} (expected one of {SUPPORTED_SOLVERS})")
    inv_temp = 1.0 / max(1e-6, float(temperature))

    def vel(state, t):
        return velocity(state, t) * inv_temp

    prev_den = None
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        if solver == "euler" or (t1 == 0.0 and solver in ("heun", "rk4")):
            x = x + dt * vel(x, t0)
        elif solver == "heun":
            k1 = vel(x, t0)
            x = x + 0.5 * dt * (k1 + vel(x + dt * k1, t1))
        elif solver == "midpoint":
            x = x + dt * vel(x + 0.5 * dt * vel(x, t0), t0 + 0.5 * dt)
        elif solver == "rk4":
            tm = t0 + 0.5 * dt
            k1 = vel(x, t0)
            k2 = vel(x + 0.5 * dt * k1, tm)
            k3 = vel(x + 0.5 * dt * k2, tm)
            k4 = vel(x + dt * k3, t1)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            den = x - t0 * vel(x, t0)  # x_data prediction
            if solver == "pingpong":
                # Denoise, then re-noise to t1 with fresh noise.
                x = (1.0 - t1) * den + t1 * torch.randn_like(den) if t1 > 0.0 else den
            else:
                # DPM-Solver++ 2M for rectified flow (as in Stable Audio's
                # sample_flow_dpmpp): second-order multistep correction in
                # log-SNR space, first order on the first and last step.
                use = den
                if prev_den is not None and t1 > 0.0:
                    h = _log_snr(t1) - _log_snr(t0)
                    r = (_log_snr(t0) - _log_snr(ts[i - 1])) / h if h != 0.0 else 0.0
                    if r != 0.0:
                        use = (1.0 + 1.0 / (2.0 * r)) * den - (1.0 / (2.0 * r)) * prev_den
                x = (t1 / t0) * x + (1.0 - t1 / t0) * use
                prev_den = den
    return x


@torch.no_grad()
def generate_chunk(
    model,
    context: torch.Tensor,
    n_steps: int,
    solver: str = "euler",
    temperature: float = 1.0,
    schedule: str = "linear",
    schedule_shift: float = 0.0,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate one chunk (`block_size` tokens) after a clean `context`.

    Both in normalized space: context [B, P, D] (P a multiple of block_size,
    may be 0) -> chunk [B, block_size, D]. The context goes through the model
    once and its per-layer K/V are cached; each solver step then runs only the
    new chunk's tokens. Exact: under block-causal attention the context never
    sees the chunk being generated.
    """
    B, P, D = context.shape
    bs = model.block_size
    x = torch.randn(B, bs, D, device=context.device, dtype=context.dtype) if noise is None else noise
    kv = model(context, context.new_zeros(B, P), return_kv=True)[1] if P > 0 else None

    def velocity(state, t):
        return model(state, state.new_full((B, bs), t), kv=kv)

    return integrate(x, build_schedule(n_steps, schedule, schedule_shift), velocity, solver, temperature)


@torch.no_grad()
def sample_joint(
    model,
    prefix: torch.Tensor,
    n_target_tokens: int,
    n_steps: int,
    solver: str = "heun",
    temperature: float = 1.0,
    schedule: str = "linear",
    schedule_shift: float = 0.0,
) -> torch.Tensor:
    """Denoise the whole continuation at once after a clean `prefix` (the
    `joint` objective's condition). Prompt + continuation must fit
    `max_seq_len`; use `generate` for longer output.

    prefix: [B, P, D] raw codec latents (P may be 0). Returns raw [B, P + T, D].
    """
    B, P, D = prefix.shape
    ctx = model.normalize(prefix)
    t_tok = torch.cat([prefix.new_zeros(B, P), prefix.new_ones(B, n_target_tokens)], dim=1)

    def velocity(state, t):
        return model(torch.cat([ctx, state], dim=1), t_tok * t)[:, P:]

    x = torch.randn(B, n_target_tokens, D, device=prefix.device, dtype=prefix.dtype)
    x = integrate(x, build_schedule(n_steps, schedule, schedule_shift), velocity, solver, temperature)
    return model.denormalize(torch.cat([ctx, x], dim=1))


@torch.no_grad()
def generate(model, prefix: torch.Tensor, n_chunks: int, context_chunks: int = 32, **kw) -> torch.Tensor:
    """Continue `prefix` chunk by chunk with a sliding context window (as realtime does).

    prefix: [B, P, D] raw codec latents (P a multiple of block_size; may be 0).
    kw: forwarded to `generate_chunk` (n_steps, solver, temperature, schedule, ...).
    Returns raw latents [B, P + n_chunks * block_size, D].
    """
    ctx = context_chunks * model.block_size
    seq = model.normalize(prefix)
    for _ in range(n_chunks):
        chunk = generate_chunk(model, seq[:, max(0, seq.shape[1] - ctx):], **kw)
        seq = torch.cat([seq, chunk], dim=1)
    return model.denormalize(seq)
