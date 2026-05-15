"""MPS-safe wrapper around `codicodec.EncoderDecoder`.

The upstream package has a few CUDA assumptions that are inconvenient on Apple
Silicon:

- `EncoderDecoder.__init__` only auto-detects CUDA (otherwise falls back to CPU).
- `codicodec.utils.distribute()` calls `torch.autocast` with
  `device_type='cuda' if device.type=='cuda' else 'cpu'`, which is wrong for
  MPS devices.
- `encoder_forward_fast` is `torch.compile`'d with `max-autotune-no-cudagraphs`
  and won't run on MPS.
- A global `mixed_precision = True` enables fp16 autocast paths that we don't
  want on MPS.

This wrapper monkey-patches the global `mixed_precision` flag to False before
constructing the model when running on a non-CUDA device, exposes
device-agnostic encode / decode helpers, and avoids the fast-path encoder
when not on CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .utils import best_device, get_logger

logger = get_logger("flow.codec_wrapper")


def _patch_codicodec_mixed_precision(value: bool) -> None:
    """Override the `mixed_precision` flag inside every relevant codicodec module.

    The `from .hparams import *` star imports copy the flag value into each
    submodule's namespace, so we patch them all.
    """
    import codicodec.hparams as _hp
    _hp.mixed_precision = value
    for modname in ("codicodec.utils", "codicodec.inference", "codicodec.models"):
        try:
            mod = __import__(modname, fromlist=["mixed_precision"])
            if hasattr(mod, "mixed_precision"):
                setattr(mod, "mixed_precision", value)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("could not patch mixed_precision in %s: %s", modname, e)


@dataclass
class CodecConfig:
    """Inference-side codec settings."""

    device: Optional[str] = None  # 'cuda' | 'mps' | 'cpu' | None=auto
    desired_channels: int = 64
    decode_mode: str = "parallel"  # 'parallel' | 'autoregressive'
    decode_steps: Optional[int] = None
    max_batch_encode: Optional[int] = None
    max_batch_decode: Optional[int] = None


class CodecWrapper:
    """Minimal, MPS-safe interface to CoDiCodec.

    Provides:
        - `encode_audio(wv, sr) -> latents [T, 8, C]`
        - `decode_latents(latents) -> waveform [2, N]`

    where C is `desired_channels` (default 64) and T is the number of latent
    timesteps. Latents are returned in the post-`atanh / sigma_rescale` space
    (approximately unit-Gaussian) ready to be fed to the flow model.
    """

    def __init__(self, cfg: CodecConfig | None = None):
        self.cfg = cfg or CodecConfig()
        self.device = best_device(self.cfg.device)
        # Patch BEFORE EncoderDecoder is constructed so the model parameters
        # are created with the right dtype expectations.
        if self.device.type != "cuda":
            _patch_codicodec_mixed_precision(False)
            logger.info("Disabled codicodec mixed_precision (running on %s)", self.device.type)

        # Defer import until we've patched the flag.
        from codicodec import EncoderDecoder

        self._encdec = EncoderDecoder(device=self.device)
        self._encdec.gen.eval()

        # Sample rate / chunk size are constants of the shipped checkpoint.
        from codicodec.hparams import sample_rate, hop, fac, spec_length, num_latents, bottleneck_channels
        self.sample_rate: int = sample_rate
        self.samples_per_chunk: int = hop * (fac // 2) * spec_length
        # The codec produces `num_latents` tokens of `bottleneck_channels` per
        # chunk, which we reshape to `tokens_per_chunk` x `desired_channels`.
        assert self.cfg.desired_channels % bottleneck_channels == 0
        self.tokens_per_chunk: int = num_latents * bottleneck_channels // self.cfg.desired_channels

        logger.info(
            "Codec ready: sr=%d, chunk=%d samples (%.3fs), tokens/chunk=%d, channels=%d",
            self.sample_rate,
            self.samples_per_chunk,
            self.samples_per_chunk / self.sample_rate,
            self.tokens_per_chunk,
            self.cfg.desired_channels,
        )

    # --------------------------------------------------------------------- #
    # Encode / Decode
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def encode_audio(self, wv: torch.Tensor | np.ndarray, sr: Optional[int] = None) -> torch.Tensor:
        """Encode a stereo or mono waveform tensor into continuous latents.

        Args:
            wv: tensor or array of shape [N] (mono), [2, N] (stereo) or
                [B, 2, N] (batch of stereo). Float32 in the [-1, 1] range.
            sr: sample rate of the input. If not None and != self.sample_rate
                the input is resampled with librosa.

        Returns:
            torch.Tensor of shape [..., T_chunks * (num_latents/desired_channels*4), desired_channels].
            For the default desired_channels=64 this is [..., T_chunks*8, 64].
        """
        if sr is not None and sr != self.sample_rate:
            wv = self._resample(wv, sr, self.sample_rate)

        latent = self._encdec.encode(
            wv,
            max_batch_size=self.cfg.max_batch_encode,
            discrete=False,
            preprocess_on_gpu=(self.device.type == "cuda"),
            desired_channels=self.cfg.desired_channels,
            fix_batch_size=False,  # avoid encoder_forward_fast on MPS
        )
        return latent

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back into a waveform.

        Accepts:
            - flat 2-D `[L, C]` (will be reshaped to `[L/8, 8, C]`),
            - 3-D `[T, 8, C]` (preferred),
            - 4-D `[B, T, 8, C]` (batched).

        Returns:
            Waveform tensor `[2, N]` (mono input becomes stereo) or `[B, 2, N]`.
        """
        if latents.dim() == 2:
            L, C = latents.shape
            assert L % self.tokens_per_chunk == 0, (
                f"latent length {L} not divisible by tokens_per_chunk={self.tokens_per_chunk}"
            )
            latents = latents.reshape(L // self.tokens_per_chunk, self.tokens_per_chunk, C)
        wv = self._encdec.decode(
            latents,
            mode=self.cfg.decode_mode,
            max_batch_size=self.cfg.max_batch_decode,
            denoising_steps=self.cfg.decode_steps,
            preprocess_on_gpu=(self.device.type == "cuda"),
        )
        return wv

    # --------------------------------------------------------------------- #
    # Streaming / live decode (`decode_next`)
    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def decode_next_chunk(self, chunk_latent: torch.Tensor) -> np.ndarray:
        """Decode a single latent chunk through CoDiCodec's live (autoregressive)
        path, maintaining the codec's internal `past_spec` / `past_latents`
        buffers across calls. Use `reset_streaming()` before starting a new
        independent stream.

        Args:
            chunk_latent: `[tokens_per_chunk, latent_dim]` or
                `[1, tokens_per_chunk, latent_dim]` (batch dim 1) in the
                codec's raw post-atanh space. For the default 64-channel
                config: shape `[8, 64]`.

        Returns:
            numpy float32 array of shape `[N]` (mono) or `[N, 2]` (stereo) for
            this chunk's decoded waveform (~32_768 samples = ~683 ms at 48 kHz
            with the shipped checkpoint).
        """
        if chunk_latent.dim() == 3:
            assert chunk_latent.shape[0] == 1, "decode_next_chunk expects a single chunk"
            chunk_latent = chunk_latent.squeeze(0)
        # codicodec's decode_next handles dtype/device placement internally
        # via `preprocess_on_gpu`. fp32 is safest across CPU/MPS/CUDA.
        chunk_latent = chunk_latent.detach().to(torch.float32)
        wv = self._encdec.decode_next(
            chunk_latent,
            preprocess_on_gpu=(self.device.type == "cuda"),
        )
        if isinstance(wv, torch.Tensor):
            wv = wv.detach().cpu().numpy()
        return np.asarray(wv, dtype=np.float32)

    def reset_streaming(self) -> None:
        """Clear the codec's internal live-decoding buffers. Required before
        starting a new independent realtime stream (or after a hard cut to
        a new seed)."""
        self._encdec.reset()

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _resample(self, wv, sr_in: int, sr_out: int):
        import librosa
        if isinstance(wv, torch.Tensor):
            wv_np = wv.detach().cpu().numpy()
        else:
            wv_np = np.asarray(wv)
        wv_rs = librosa.resample(wv_np, orig_sr=sr_in, target_sr=sr_out, axis=-1)
        return torch.from_numpy(wv_rs)
