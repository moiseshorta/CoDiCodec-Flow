"""Block-causal flow-matching DiT.

Operates on flattened CoDiCodec continuous latents
`x ∈ R^{B x L x latent_dim}` where every consecutive `block_size` tokens
correspond to one ~0.683 s audio chunk.

Attention is **block-causal**: full bidirectional attention within a block,
strict causality across blocks. This matches the codec's chunk-causal
decoder and lets us KV-cache one chunk at a time at inference.

Time conditioning is **per-token**: this is essential because in a continuation
training step the prefix tokens are clean (t=0) while the target tokens carry
a noise level `t ∈ (0, 1]`. AdaLN-Zero modulation handles this cleanly.

Architectural choices follow recent SOTA continuous-token diffusion / flow
matching models (DiT, SD3, Stable Audio 2):

- Pre-norm RMSNorm + AdaLN-Zero modulation (DiT).
- QK-norm before scaled dot-product attention (SD3).
- SwiGLU MLP.
- **Rotary positional embeddings (RoPE)** on Q/K, with a factorized
  intra-chunk / inter-chunk frequency split so the model can distinguish
  within-block from across-block positions cleanly.
- **Latent normalization** baked in as buffers: `cfm_loss` and `sample`
  call `model.normalize` / `model.denormalize` so the network always sees
  a unit-Gaussian-distributed input regardless of the dataset's natural
  scale, and downloaded checkpoints are self-contained.
- Optional residual dropout for regularization on small datasets.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Rotary positional embeddings (factorized intra-chunk / inter-chunk)
# --------------------------------------------------------------------------- #

class FactorizedRoPE(nn.Module):
    """Rotary positional embedding split between intra-chunk and inter-chunk.

    We split the head dimension in half:

        - the **first half** rotates by *intra-chunk* position
          (i.e. position within a `block_size`-sized chunk),
        - the **second half** rotates by *inter-chunk* position
          (i.e. which chunk in the sequence we are at).

    This matches the natural axes of the data (a chunk is one ~0.7 s audio
    frame, the chunk index is "time across chunks") while still using a
    single monolithic attention. Conceptually similar to the 2D-axis RoPE
    used in SD3 / Hunyuan-DiT for image patches.
    """

    def __init__(self, head_dim: int, block_size: int, max_chunks: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 4 == 0, (
            f"head_dim={head_dim} must be a multiple of 4 (split into two halves of pairs)"
        )
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_chunks = max_chunks
        max_seq_len = block_size * max_chunks

        per_half = head_dim // 2
        n_freqs = per_half // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, n_freqs, dtype=torch.float32) / n_freqs))

        idx = torch.arange(max_seq_len, dtype=torch.float32)
        intra_pos = idx % block_size
        inter_pos = torch.div(idx, block_size, rounding_mode="floor")

        ang_intra = intra_pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
        ang_inter = inter_pos.unsqueeze(-1) * inv_freq.unsqueeze(0)

        cos_intra = ang_intra.cos().repeat_interleave(2, dim=-1)
        sin_intra = ang_intra.sin().repeat_interleave(2, dim=-1)
        cos_inter = ang_inter.cos().repeat_interleave(2, dim=-1)
        sin_inter = ang_inter.sin().repeat_interleave(2, dim=-1)

        cos = torch.cat([cos_intra, cos_inter], dim=-1)  # [L, head_dim]
        sin = torch.cat([sin_intra, sin_inter], dim=-1)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.stack((-x2, x1), dim=-1)
        return out.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, head_dim]
        n = x.shape[-2]
        if n > self.cos.shape[0]:
            raise ValueError(f"sequence length {n} exceeds RoPE max ({self.cos.shape[0]})")
        cos = self.cos[:n].to(x.dtype)
        sin = self.sin[:n].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _zero_(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def _xavier_(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding for a per-token noise level t ∈ [0, 1].

    Args:
        t: [..., L] in [0, 1].
        dim: output embedding dimension (must be even).
    Returns:
        Tensor [..., L, dim].
    """
    assert dim % 2 == 0, "embedding dim must be even"
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
    )
    # Scale t by 1000 so the network sees enough variation across [0, 1].
    args = t.unsqueeze(-1).float() * freqs * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.to(t.dtype if t.is_floating_point() else torch.float32)


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """RMSNorm with no affine (modulation comes from AdaLN-Zero)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float for numerical stability on MPS.
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).type_as(x)


# --------------------------------------------------------------------------- #
# Attention & MLP
# --------------------------------------------------------------------------- #

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        rope: Optional[FactorizedRoPE] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim == n_heads * head_dim, (
            f"dim ({dim}) must equal n_heads * head_dim ({n_heads}*{head_dim})"
        )
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv = _xavier_(nn.Linear(dim, dim * 3, bias=False))
        # NB: do NOT zero-init `out`. Combined with the AdaLN-Zero gate this
        # would zero out `∂L/∂gate_a` (since `gate_a * attn(...) = 0` and the
        # gradient w.r.t. the gate is `∂L/∂residual * attn(...) = 0`). The
        # AdaLN gate alone is sufficient for the residual to start as identity.
        self.out = _xavier_(nn.Linear(dim, dim, bias=False))
        # Per-head q/k normalization stabilizes training (used in many recent DiTs).
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = rope
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to [B, H, N, D]
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # Rotary embeddings (V is NOT rotated -- standard RoPE convention).
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.out(out)


class SwiGLU(nn.Module):
    """Gated MLP: (silu(W1 x) * W2 x) @ W3."""

    def __init__(self, dim: int, mlp_mult: int, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * mlp_mult)
        # We split the first linear in half so the inner state is `inner//2`.
        # This keeps the param count comparable to a standard MLP w/ gelu.
        self.w1 = _xavier_(nn.Linear(dim, inner, bias=False))
        # NB: do NOT zero-init `w2`. See SelfAttention.out comment: pairing two
        # zero-init layers in series with an AdaLN-Zero gate kills the gate's
        # gradient and the block can never become active.
        self.w2 = _xavier_(nn.Linear(inner // 2, dim, bias=False))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.dropout(F.silu(a) * b))


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #

class DiTBlock(nn.Module):
    """DiT transformer block with AdaLN-Zero conditioning, per-token cond.

    Conditioning is `[B, L, cond_dim]`; the block produces 6 modulation
    parameters per token (scale_attn, shift_attn, gate_attn, scale_mlp,
    shift_mlp, gate_mlp). All six are zero-initialized via the modulation
    projection so the block starts as identity (gates = 0).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        mlp_mult: int,
        cond_dim: int,
        rope: Optional[FactorizedRoPE] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_heads, head_dim, rope=rope, dropout=dropout)
        self.norm_mlp = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_mult=mlp_mult, dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # 6 * dim of modulation params per token.
        self.mod = _zero_(nn.Linear(cond_dim, 6 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # c: [B, L, cond_dim]
        mod = self.mod(F.silu(c))  # [B, L, 6*dim]
        scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = mod.chunk(6, dim=-1)

        # Attention path
        h = self.norm_attn(x) * (1.0 + scale_a) + shift_a
        h = self.attn(h, attn_mask=attn_mask)
        x = x + gate_a * self.resid_dropout(h)

        # MLP path
        h = self.norm_mlp(x) * (1.0 + scale_m) + shift_m
        h = self.mlp(h)
        x = x + gate_m * self.resid_dropout(h)
        return x


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class FlowDiT(nn.Module):
    """Block-causal flow-matching DiT for CoDiCodec latents.

    Forward signature:
        v_pred = model(x_t, t, attn_mask=None)
    where:
        x_t: [B, L, latent_dim]  -- in **normalized** space
                                    (caller applies `model.normalize`).
        t:   [B, L] noise level in [0, 1] per token.
    Predicts the velocity field `v = ε - x_data` in normalized space.
    `cfm_loss` and `sample()` handle (de)normalization transparently.
    """

    def __init__(
        self,
        latent_dim: int,
        block_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        mlp_mult: int,
        cond_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
        lat_mean: Optional[torch.Tensor] = None,
        lat_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.dim = dim
        self.cond_dim = cond_dim
        self.max_seq_len = max_seq_len

        # Latent normalization stats baked into the checkpoint as buffers so
        # downloaded weights are self-contained. Default to identity if no
        # stats are passed; train.py overwrites them with dataset stats.
        if lat_mean is None:
            lat_mean = torch.zeros(latent_dim, dtype=torch.float32)
        if lat_std is None:
            lat_std = torch.ones(latent_dim, dtype=torch.float32)
        self.register_buffer("lat_mean", lat_mean.float().clone())
        self.register_buffer("lat_std", lat_std.float().clamp(min=1e-3).clone())

        # Input projection
        self.proj_in = _xavier_(nn.Linear(latent_dim, dim))

        # Rotary positional embeddings, factorized intra/inter-chunk.
        max_chunks = max_seq_len // block_size
        self.rope = FactorizedRoPE(head_dim=head_dim, block_size=block_size, max_chunks=max_chunks)

        # Time conditioning: sinusoidal -> 2-layer MLP -> per-token cond vector.
        # Standard DiT/SD3 t-embedder ends with Linear (no trailing activation):
        # the SiLU is applied inside each block's `mod` projection.
        self.cond_mlp = nn.Sequential(
            _xavier_(nn.Linear(cond_dim, cond_dim)),
            nn.SiLU(),
            _xavier_(nn.Linear(cond_dim, cond_dim)),
        )

        # Transformer stack
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    n_heads=n_heads,
                    head_dim=head_dim,
                    mlp_mult=mlp_mult,
                    cond_dim=cond_dim,
                    rope=self.rope,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Output: AdaLN-Zero modulation -> Linear (zero init) so v_pred starts at 0.
        self.norm_out = RMSNorm(dim)
        self.mod_out = _zero_(nn.Linear(cond_dim, 2 * dim))
        self.proj_out = _zero_(nn.Linear(dim, latent_dim))

    # ----- normalization helpers ------------------------------------------- #

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Raw codec latents -> unit-Gaussian space."""
        return (x - self.lat_mean) / self.lat_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Unit-Gaussian space -> raw codec latents."""
        return x * self.lat_std + self.lat_mean

    def set_lat_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.lat_mean.copy_(mean.to(self.lat_mean.dtype))
        self.lat_std.copy_(std.to(self.lat_std.dtype).clamp(min=1e-3))

    # ----- forward ---------------------------------------------------------- #

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict velocity at noise level `t` (in normalized space).

        Args:
            x_t: [B, L, latent_dim] in *normalized* space.
            t:   [B, L] floats in [0, 1].
            attn_mask: optional [L, L] additive mask. If None, a block-causal
                       mask is built on the fly.
        """
        b, l, _ = x_t.shape
        if l > self.max_seq_len:
            raise ValueError(
                f"sequence length {l} exceeds max_seq_len {self.max_seq_len}"
            )
        h = self.proj_in(x_t)

        # Per-token conditioning.
        t_emb = timestep_embedding(t, self.cond_dim)  # [B, L, cond_dim]
        c = self.cond_mlp(t_emb)

        if attn_mask is None:
            from ..utils import block_causal_mask
            attn_mask = block_causal_mask(l, self.block_size, device=h.device)
        # Cast to the activation dtype so SDPA doesn't see mixed dtypes inside autocast.
        attn_mask = attn_mask.to(h.dtype)

        for block in self.blocks:
            h = block(h, c, attn_mask=attn_mask)

        # Output AdaLN + projection
        scale, shift = self.mod_out(F.silu(c)).chunk(2, dim=-1)
        h = self.norm_out(h) * (1.0 + scale) + shift
        v_pred = self.proj_out(h)
        return v_pred
