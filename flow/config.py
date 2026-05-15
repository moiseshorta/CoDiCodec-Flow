"""Configuration dataclasses for `codicodec-flow`.

Kept dependency-free (no hydra/omegaconf) so the project remains easy to run
on a personal Mac. CLI entry points instantiate these dataclasses and apply
overrides parsed from argparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Constants of the shipped CoDiCodec checkpoint. Hardcoded so config is
# self-contained, but verified against `codicodec.hparams` in `codec_wrapper`.
CODEC_SAMPLE_RATE = 48_000
CODEC_CHUNK_SAMPLES = 32_768           # 32 STFT frames * 1024 hop * (fac/2=1)
CODEC_TOKENS_PER_CHUNK = 8             # for desired_channels=64
CODEC_LATENT_DIM = 64                  # for desired_channels=64
CODEC_FRAME_RATE_HZ = CODEC_SAMPLE_RATE / CODEC_CHUNK_SAMPLES * CODEC_TOKENS_PER_CHUNK


@dataclass
class ModelConfig:
    """Block-causal Flow-Matching DiT.

    Defaults target ~100M params (~97M precisely): dim=768, depth=10, 12 heads
    of size 64. Designed to fit on a 36 GB Apple Silicon machine with
    `batch_size=4`, `crop_tokens=512`, bf16 autocast.

    Tip: for small datasets (~30 min of audio, 1-5 shards) lower `dim` to 384
    and `n_layers` to 6 -- this drops to ~12M params and matches dataset
    capacity, which generalizes much better than the full 100M.
    """

    latent_dim: int = CODEC_LATENT_DIM        # input/output channels
    block_size: int = CODEC_TOKENS_PER_CHUNK  # tokens per chunk (intra-block bidirectional)

    dim: int = 768
    n_layers: int = 10
    n_heads: int = 12
    head_dim: int = 64
    mlp_mult: int = 4
    # Dropout applied to MLP hidden states + attention residuals. 0.1 is a
    # safe default for moderate-size datasets; raise to 0.15-0.2 for tiny
    # datasets, drop to 0.0 for very large pretraining runs.
    dropout: float = 0.1

    # Maximum sequence length the model can handle (in tokens, NOT chunks).
    # 1024 tokens = 128 chunks ≈ 87 seconds of audio.
    max_seq_len: int = 1024

    # Time conditioning: dimensionality of the sigma embedding used in AdaLN.
    # Kept equal to `dim` so the per-token modulation MLP is well-matched.
    cond_dim: int = 768


@dataclass
class CFMConfig:
    """Conditional Flow Matching training hyperparameters."""

    # Time sampling.
    #   'logit_normal' -- SD3-style t = sigmoid(N(loc, scale)). Concentrates
    #                     mass near 0.5 with light tails. Best for large
    #                     datasets.
    #   'uniform'      -- t ~ Uniform(0, 1). Better for small datasets where
    #                     coverage of the [0, 0.1] tail (the hard "final
    #                     denoising steps") matters more than concentration.
    t_sample_mode: str = "logit_normal"
    logit_normal_loc: float = 0.0
    logit_normal_scale: float = 1.0

    # Min-SNR-γ loss weighting (Hang et al. 2023). 0 / None disables; 5 is the
    # paper's default and a good starting point. For v-prediction with linear
    # rectified flow this is a relatively small effect, so we keep it off by
    # default but expose the knob.
    min_snr_gamma: float = 0.0

    # Probability of treating the entire sample as a "fresh start" with no
    # prefix (full-sequence denoising). Helps the model learn unconditional
    # generation alongside continuation.
    p_unconditional: float = 0.1

    # Range of prefix length, expressed as a fraction of the full sequence.
    prefix_min_frac: float = 0.05
    prefix_max_frac: float = 0.75

    # Sampling defaults.
    sample_nfe: int = 8
    sample_solver: str = "heun"  # 'euler' | 'heun'


@dataclass
class DataConfig:
    """Latent-dataset settings."""

    data_dir: str = "./data/latents"

    # Training crop length, in latent tokens (1 chunk = block_size tokens).
    # 768 tokens = 96 chunks ≈ 65.5 seconds @ 48 kHz.  (Codec produces
    # 8 tokens / 0.683 s; 96 chunks * 0.683 s = 65.5 s.)
    crop_tokens: int = 768

    val_frac: float = 0.05
    seed: int = 42
    num_workers: int = 2  # MPS does not benefit much from many workers


@dataclass
class TrainConfig:
    """Optimization & training-loop settings tuned for a single MPS device."""

    out_dir: str = "./runs/v0"

    # Optional checkpoint to initialize *weights* from when out_dir has no
    # last.pt (fine-tuning workflow). Optimizer state and step counter are
    # NOT loaded; lat_mean/lat_std buffers are kept fresh from the new
    # dataset rather than overwritten by the source checkpoint. Auto-resume
    # from out_dir/last.pt takes precedence if it exists.
    init_from: Optional[str] = None

    batch_size: int = 8
    grad_accum: int = 2
    max_steps: int = 200_000
    log_every: int = 50
    val_every: int = 1_000
    ckpt_every: int = 5_000

    # Periodic audio sampling so the user can hear training progress.
    audio_sample_every: int = 1_000     # set to 0 to disable
    audio_n_samples: int = 2            # number of distinct samples per call
    # If True (default), generate unconditional audio from pure noise (no
    # prefix). If False, take a `audio_prompt_seconds` clip from the val set
    # and ask the model to continue it for `audio_continuation_seconds`.
    audio_unconditional: bool = True
    audio_prompt_seconds: float = 4.0   # length of clean prompt (continuation mode only)
    audio_continuation_seconds: float = 8.0  # length of generated audio (also used for total length in unconditional mode together with prompt_seconds)
    audio_nfe: int = 8                  # ODE steps for sampling
    audio_solver: str = "heun"          # 'euler' | 'heun'
    audio_use_ema: bool = True          # generate with EMA weights (preferred)

    lr: float = 1e-4
    warmup_steps: int = 2_000
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    ema_decay: float = 0.9999

    # Mixed precision settings. bf16 generally works on MPS but coverage is
    # uneven; fp32 is the safe fallback.
    dtype: str = "bf16"  # 'fp32' | 'bf16' | 'fp16'

    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    cfm: CFMConfig = field(default_factory=CFMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: Optional[str] = None  # None = auto-detect
