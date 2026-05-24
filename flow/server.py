"""Realtime FlowDiT + CoDiCodec generation server (stdio JSON IPC).

Wraps `RealtimeFlowGenerator` with:
  - non-blocking JSON-line command parser on stdin
  - per-chunk JSON-line state events on stdout
  - per-parameter LFO engine (8 LFOs, evaluated each chunk)
  - audio plays from this process via sounddevice (already faster-than-realtime)

Designed to be spawned by an Electron front-end as a subprocess.

Wire format (newline-delimited JSON, one object per line):

  Commands (Electron -> Python, stdin):
    {"type":"set_param","name":"temperature","value":1.2}
    {"type":"set_param","name":"n_steps","value":8}
    {"type":"set_param","name":"solver","value":"heun"}
    {"type":"set_param","name":"context_chunks","value":24}
    {"type":"set_param","name":"crossfade_chunks","value":4}
    {"type":"set_summary_token","idx":3,"kind":"scale","value":1.5}
    {"type":"set_channel_dim","idx":17,"kind":"bias","value":0.4}
    {"type":"set_summary_vector","kind":"scale","values":[...8 floats...]}
    {"type":"set_channel_vector","kind":"bias","values":[...64 floats...]}
    {"type":"randomize","group":"summary","kind":"both"}
    {"type":"randomize","group":"channel","kind":"scale"}
    {"type":"reset","group":"summary"}
    {"type":"reset","group":"channel"}
    {"type":"reset","group":"all"}
    {"type":"new_seed","mode":"crossfade","seed":12345}    # mode: crossfade|hardcut|random
    {"type":"set_lfo","name":"temperature","enabled":true,
     "rate":0.4,"depth":0.3,"shape":"sine","phase":0.0}
    {"type":"apply_state","params":{"temperature":1.1,...},
     "summary_scale":[...8],"summary_bias":[...8],
     "channel_scale":[...64],"channel_bias":[...64],
     "seed":12345,"seed_mode":"crossfade"}
    {"type":"shutdown"}

  Events (Python -> Electron, stdout):
    {"type":"ready", ...config snapshot...}
    {"type":"chunk", ...per-chunk state + last latent chunk...}
    {"type":"log","level":"info","msg":"..."}
    {"type":"error","msg":"..."}

Run standalone for debugging:
    python -m flow.server --ckpt ./runs/.../last.pt --device mps
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import select
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None

from .codec_wrapper import CodecConfig, CodecWrapper
from .realtime import RealtimeFlowGenerator, load_model
from .utils import best_device, get_logger

# Redirect every existing `flow.*` log handler to stderr so the stdout channel
# stays a clean JSON-line stream for the Electron front-end. Must run AFTER the
# imports above because each `flow.*` module installs its own stdout handler at
# import time via `get_logger()`.
def _redirect_logs_to_stderr() -> None:
    import logging
    seen = set()
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if not isinstance(lg, logging.Logger):
            continue
        for h in lg.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout:
                h.stream = sys.stderr
                seen.add(name)
    return seen


_redirect_logs_to_stderr()
logger = get_logger("flow.server")
# `get_logger` may have just installed a fresh stdout handler for `flow.server`;
# re-route it too.
_redirect_logs_to_stderr()


# --------------------------------------------------------------------------- #
# LFO engine
# --------------------------------------------------------------------------- #

LFO_SHAPES = ("sine", "triangle", "saw", "square", "random")


class LFO:
    """Single low-frequency oscillator with bipolar output centered at 0.

    `value(t_s)` returns a float in roughly `[-depth, +depth]`. Caller adds it
    to the static base (so center=0 means "modulate around the base value").
    """

    __slots__ = ("enabled", "rate_hz", "depth", "shape", "phase",
                 "_sh_last_t", "_sh_value")

    def __init__(self) -> None:
        self.enabled: bool = False
        self.rate_hz: float = 0.5
        self.depth: float = 0.3
        self.shape: str = "sine"
        self.phase: float = 0.0
        # Sample-and-hold state
        self._sh_last_t: float = -1.0
        self._sh_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rate": float(self.rate_hz),
            "depth": float(self.depth),
            "shape": self.shape,
            "phase": float(self.phase),
        }

    def update(self, payload: Dict[str, Any]) -> None:
        if "enabled" in payload:
            self.enabled = bool(payload["enabled"])
        if "rate" in payload:
            self.rate_hz = max(0.0, float(payload["rate"]))
        if "depth" in payload:
            self.depth = max(0.0, float(payload["depth"]))
        if "shape" in payload:
            sh = str(payload["shape"]).lower()
            if sh in LFO_SHAPES:
                self.shape = sh
        if "phase" in payload:
            self.phase = float(payload["phase"]) % 1.0

    def value(self, t_s: float) -> float:
        if not self.enabled or self.depth <= 0.0 or self.rate_hz <= 0.0:
            return 0.0
        x = (t_s * self.rate_hz + self.phase) % 1.0
        sh = self.shape
        if sh == "sine":
            v = math.sin(2.0 * math.pi * x)
        elif sh == "triangle":
            v = 4.0 * abs(x - 0.5) - 1.0
        elif sh == "saw":
            v = 2.0 * x - 1.0
        elif sh == "square":
            v = 1.0 if x < 0.5 else -1.0
        elif sh == "random":
            # one new value per period
            period = 1.0 / max(self.rate_hz, 1e-6)
            slot = math.floor(t_s / period)
            if slot != self._sh_last_t:
                self._sh_last_t = slot
                self._sh_value = (random.random() * 2.0 - 1.0)
            v = self._sh_value
        else:
            v = 0.0
        return float(v) * float(self.depth)


# Names of the LFO-enabled parameter groups (frontend & backend agree on these).
LFO_NAMES = (
    "temperature",
    "n_steps",
    "context_chunks",
    "crossfade_chunks",
    "summary_scale",
    "summary_bias",
    "channel_scale",
    "channel_bias",
)


# --------------------------------------------------------------------------- #
# Live audio-input conditioner                                                 #
#                                                                              #
# Lets the user "improvise" with the model: their microphone audio is encoded  #
# by the codec into the model's normalized latent space, and used as the       #
# conditioning prefix for the next generated chunk(s). The encoder runs in a  #
# dedicated worker thread so the audio output thread is never blocked.        #
# --------------------------------------------------------------------------- #


class InputAudioConditioner:
    """Stream mic audio in, encode it on a worker thread, blend into prefix.

    Threading model:
      - `push_samples()`  is called from the IPC dispatch (audio loop thread)
        whenever a new mic chunk arrives. It just appends mono float32 samples
        to a small ring buffer and signals the worker.
      - `_worker_loop()`  runs in its own daemon thread. When at least
        `samples_per_chunk` fresh samples are available it:
            1. pops the most recent `samples_per_chunk` samples (mono),
            2. duplicates to stereo and runs `codec.encode_audio()`,
            3. normalizes via `model.normalize()`,
            4. atomically replaces `self._latest_norm`.
      - `apply_to_prefix()`  is called by the audio loop just before each
        generation step. If `enabled` and a fresh latent is available it
        blends the latest mic latent into `gen.prefix_norm`.

    The encode step takes ~80-200 ms on MPS for ~683 ms of audio (well under
    real-time), so the worker keeps pace with mic input naturally.
    """

    def __init__(self, gen, *, ring_seconds: float = 4.0) -> None:
        self.gen = gen
        self.codec = gen.codec
        self.model = gen.model
        self.device = gen.device
        self.dtype = gen.dtype
        self.sample_rate = int(gen.sample_rate)
        self.samples_per_chunk = int(gen.samples_per_chunk)
        self.block_size = int(gen.block_size)
        self.latent_dim = int(gen.latent_dim)

        # User-tunable settings (all atomic-by-assignment; floats/bools).
        self.enabled: bool = False
        self.input_gain: float = 1.0
        # Blend factor in [0, 1]: 0 = pure self-continuation (ignore mic),
        # 1 = replace last `block_size` prefix tokens with mic-encoded ones.
        # Intermediate values lerp in normalized latent space.
        self.blend: float = 0.85

        ring_n = max(self.samples_per_chunk * 2, int(ring_seconds * self.sample_rate))
        self._ring = np.zeros(ring_n, dtype=np.float32)
        self._ring_n = ring_n
        self._write_idx = 0
        self._fresh_samples = 0  # samples since last successful encode
        self._lock = threading.Lock()
        self._wake = threading.Event()

        # Latest encoded mic chunk, in the model's normalized space:
        # shape `[1, block_size, latent_dim]`, dtype = self.dtype, on device.
        self._latest_norm: Optional[torch.Tensor] = None
        self._latest_norm_lock = threading.Lock()

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop,
                                        name="input-conditioner",
                                        daemon=True)
        self._thread.start()

    # ----- ingestion (called from IPC thread) ----- #

    def push_samples(self, mono: np.ndarray, src_sr: int) -> None:
        """Append a chunk of mono float32 samples to the ring buffer.

        If `src_sr != self.sample_rate` we resample with a fast linear
        interpolator (good enough for live conditioning; the codec is the
        spectral arbiter anyway).
        """
        if mono.dtype != np.float32:
            mono = mono.astype(np.float32, copy=False)
        if mono.ndim != 1:
            mono = mono.reshape(-1)
        if mono.size == 0:
            return
        if self.input_gain != 1.0:
            mono = mono * float(self.input_gain)
        if src_sr != self.sample_rate and src_sr > 0:
            # Linear resample. Fine for ~480ms of mic at a time.
            n_out = int(round(mono.size * self.sample_rate / src_sr))
            if n_out < 1:
                return
            xp = np.arange(mono.size, dtype=np.float32)
            x = np.linspace(0.0, mono.size - 1, n_out, dtype=np.float32)
            mono = np.interp(x, xp, mono).astype(np.float32, copy=False)

        with self._lock:
            n = mono.size
            ring = self._ring
            rn = self._ring_n
            i = self._write_idx
            end = i + n
            if end <= rn:
                ring[i:end] = mono
            else:
                first = rn - i
                ring[i:] = mono[:first]
                ring[: end - rn] = mono[first:]
            self._write_idx = end % rn
            self._fresh_samples = min(self._fresh_samples + n, rn)
        # Signal worker that new audio is available.
        self._wake.set()

    def _take_recent(self, n: int) -> Optional[np.ndarray]:
        """Atomically copy the last `n` samples written to the ring buffer.

        Returns None if fewer than `n` samples have ever been written.
        """
        with self._lock:
            if self._fresh_samples < n:
                return None
            ring = self._ring
            rn = self._ring_n
            i = self._write_idx
            start = (i - n) % rn
            if start + n <= rn:
                out = ring[start: start + n].copy()
            else:
                first = rn - start
                out = np.empty(n, dtype=np.float32)
                out[:first] = ring[start:]
                out[first:] = ring[: n - first]
            self._fresh_samples = 0
            return out

    # ----- encoder worker ----- #

    def _worker_loop(self) -> None:
        try:
            while not self._stop.is_set():
                # Wait until ingestion signals new audio (or shutdown).
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                if self._stop.is_set():
                    return
                if not self.enabled:
                    continue
                samples = self._take_recent(self.samples_per_chunk)
                if samples is None:
                    continue
                try:
                    norm = self._encode_to_norm(samples)
                except Exception as e:
                    # Don't crash the engine on a bad mic chunk; just log.
                    print(f"[input] encode failed: {e}", file=sys.stderr)
                    continue
                with self._latest_norm_lock:
                    self._latest_norm = norm
        except Exception as e:
            print(f"[input] worker exited: {e}", file=sys.stderr)

    @torch.no_grad()
    def _encode_to_norm(self, mono: np.ndarray) -> torch.Tensor:
        """Mono float32 samples -> [1, block_size, latent_dim] normalized."""
        # Make stereo by duplicating (codec expects stereo input).
        wv = np.stack([mono, mono], axis=0)  # [2, N]
        # IMPORTANT: keep the waveform on CPU. CoDiCodec's encode pipeline
        # only does its preprocessing (mel-spectrogram + windowing) on GPU
        # when CUDA is available. On MPS / CPU it expects the waveform on
        # CPU and moves the result internally. Handing it an MPS tensor
        # would crash with "tensors on different devices" inside the codec.
        wv_t = torch.from_numpy(wv)
        latent = self.codec.encode_audio(wv_t, sr=self.sample_rate)
        # Make sure the latent ends up on the model's device for normalize().
        latent = latent.to(self.device)
        if latent.dim() == 2:
            latent = latent.reshape(-1, self.block_size, latent.shape[-1])  # [T, 8, 64]
        elif latent.dim() == 4:
            latent = latent.squeeze(0)  # [T, 8, 64]
        # Take the last block_size tokens (one chunk in the codec's raw post-atanh space).
        chunk_raw = latent[-1:].to(dtype=self.dtype)  # [1, block_size, latent_dim]
        chunk_norm = self.model.normalize(chunk_raw)
        return chunk_norm

    # ----- prefix injection (called from audio loop) ----- #

    def apply_to_prefix(self) -> bool:
        """Blend the latest mic latent into `gen.prefix_norm`. Returns True
        if any modification happened (used for telemetry).

        Strategy: replace the last `block_size` tokens of the prefix with
        a lerp between (current prefix tail) and (mic latent), weighted by
        `self.blend`. If the prefix is shorter than `block_size`, append.
        """
        if not self.enabled or self.blend <= 0.0:
            return False
        with self._latest_norm_lock:
            mic = self._latest_norm
        if mic is None:
            return False

        gen = self.gen
        pref = gen.prefix_norm
        n = mic.shape[1]
        a = float(max(0.0, min(1.0, self.blend)))
        if pref.shape[1] >= n:
            tail = pref[:, -n:, :]
            blended = tail * (1.0 - a) + mic.to(device=tail.device, dtype=tail.dtype) * a
            new_prefix = torch.cat([pref[:, :-n, :], blended], dim=1)
        else:
            # Prefix shorter than one chunk: just append the mic chunk.
            new_prefix = torch.cat(
                [pref, mic.to(device=pref.device, dtype=pref.dtype)], dim=1
            )
        gen.prefix_norm = new_prefix
        return True

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()


# --------------------------------------------------------------------------- #
# Audio device helpers (sounddevice)
# --------------------------------------------------------------------------- #


def _list_output_devices() -> "list[Dict[str, Any]]":
    """Return all sounddevice devices that have at least one output channel.

    Each entry is a JSON-safe dict: {index, name, max_output_channels,
    default_samplerate, hostapi_name, is_default}.
    """
    if sd is None:
        return []
    out: "list[Dict[str, Any]]" = []
    try:
        # `sd.default.device` is a (input, output) pair-like object that
        # supports indexing but is *not* a list/tuple, so probe via try/except.
        default_idx = -1
        try:
            v = sd.default.device[1]
            if v is not None:
                default_idx = int(v)
        except Exception:
            pass
        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            if int(d.get("max_output_channels", 0)) <= 0:
                continue
            hi = int(d.get("hostapi", 0))
            hostapi_name = hostapis[hi]["name"] if 0 <= hi < len(hostapis) else ""
            out.append({
                "index": int(i),
                "name": str(d.get("name", f"device {i}")),
                "max_output_channels": int(d.get("max_output_channels", 0)),
                "default_samplerate": float(d.get("default_samplerate", 0.0) or 0.0),
                "hostapi_name": hostapi_name,
                "is_default": (i == default_idx),
            })
    except Exception:  # pragma: no cover
        pass
    return out


def _device_supports_stereo_output(idx: Optional[int]) -> bool:
    """Return True iff the given sounddevice index has >= 2 output channels.

    `None` (system default) is always considered acceptable: the OS will pick
    a sensible output. `sd` being unavailable counts as "unknown -> accept".
    """
    if idx is None:
        return True
    if sd is None:
        return True
    try:
        info = sd.query_devices(int(idx))
        return int(info.get("max_output_channels", 0)) >= 2
    except Exception:
        return False


def _resolve_output_device(spec: Optional[Any]) -> Optional[int]:
    """Resolve a user-supplied output-device spec to a concrete int index.

    Accepts:
      - None / "" / "default"  -> system default (return None)
      - int                    -> verbatim index
      - str numeric            -> int(spec)
      - str                    -> first device whose name contains `spec`
                                  (case-insensitive)

    Returns the chosen device index, or None to mean "use the system default".
    On any failure to match, OR if the resolved device cannot do stereo
    output, falls back to None and logs a warning to stderr. This prevents
    a stale saved preference (e.g. an unplugged interface) from crashing
    the engine at startup with `PaErrorCode -9998`.
    """
    if spec is None:
        return None

    resolved: Optional[int] = None
    if isinstance(spec, int):
        resolved = int(spec)
    elif isinstance(spec, str):
        s = spec.strip()
        if not s or s.lower() == "default":
            return None
        try:
            resolved = int(s)
        except ValueError:
            resolved = None
        if resolved is None:
            if sd is None:
                return None
            try:
                needle = s.lower()
                for i, d in enumerate(sd.query_devices()):
                    if int(d.get("max_output_channels", 0)) <= 0:
                        continue
                    if needle in str(d.get("name", "")).lower():
                        resolved = int(i)
                        break
            except Exception:
                pass
            if resolved is None:
                sys.stderr.write(f"[server] output-device '{spec}' not found, using default\n")
                return None

    if resolved is not None and not _device_supports_stereo_output(resolved):
        sys.stderr.write(
            f"[server] output-device {resolved!r} does not support stereo output, "
            f"using system default\n"
        )
        return None
    return resolved


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


class ServerApp:
    """Owns the generator, LFOs, and the stdio IPC loop."""

    def __init__(self, gen: RealtimeFlowGenerator, *, max_chunks: Optional[int] = None,
                 save_path: Optional[str] = None,
                 output_device: Optional[Any] = None) -> None:
        self.gen = gen
        self.max_chunks = max_chunks
        self.save_path = save_path
        # Resolved output device (int index for sounddevice, or None for default).
        self._output_device: Optional[int] = _resolve_output_device(output_device)
        self.lfos: Dict[str, LFO] = {n: LFO() for n in LFO_NAMES}

        # Snapshot of static base values (last user-set or programmatic value).
        # The generator's live attributes are recomputed each chunk as
        # `static + lfo.value()` whenever the LFO is enabled. When the LFO is
        # disabled we restore the static value as the live one.
        self._base_temperature: float = float(gen.temperature)
        self._base_n_steps: int = int(gen.n_steps)
        self._base_context_chunks: int = int(gen.context_chunks)
        self._base_crossfade_chunks: int = int(gen.crossfade_chunks)
        # Static vectors live on the generator already (summary_scale/bias,
        # channel_scale/bias). We keep our own copies so we can restore them
        # cleanly each chunk before applying any vector LFO offset.
        self._base_summary_scale: torch.Tensor = gen.summary_scale.detach().clone()
        self._base_summary_bias: torch.Tensor = gen.summary_bias.detach().clone()
        self._base_channel_scale: torch.Tensor = gen.channel_scale.detach().clone()
        self._base_channel_bias: torch.Tensor = gen.channel_bias.detach().clone()

        # Master output gain (applied in the audio callback). 1.0 = unity.
        self._master_gain: float = 1.0

        # Live audio-input conditioner (mic -> latent prefix). Always
        # constructed, but only "active" when self.input.enabled is True.
        self.input = InputAudioConditioner(gen)

        self._t_start: float = time.time()
        self._stop_flag = threading.Event()
        self._stdin_lock = threading.Lock()
        self._cmd_queue: "list[Dict[str, Any]]" = []
        # Serialize stdout writes across threads (chunk events + waveform pusher).
        self._emit_lock = threading.Lock()

        # Use a daemon thread to read stdin so we never block the audio loop.
        self._reader = threading.Thread(target=self._stdin_reader, daemon=True)
        self._reader.start()

    # ----- IPC helpers ----- #

    def _emit(self, obj: Dict[str, Any]) -> None:
        """Write a single JSON line to stdout. Best-effort; ignores broken pipe."""
        try:
            line = json.dumps(obj, separators=(",", ":")) + "\n"
            with self._emit_lock:
                sys.stdout.write(line)
                sys.stdout.flush()
        except (BrokenPipeError, OSError):
            self._stop_flag.set()

    def _log(self, level: str, msg: str) -> None:
        self._emit({"type": "log", "level": level, "msg": msg})

    def _stdin_reader(self) -> None:
        """Background thread: parse one JSON object per line, queue it."""
        try:
            for raw in sys.stdin:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as e:
                    self._log("error", f"bad json: {e}")
                    continue
                if not isinstance(obj, dict) or "type" not in obj:
                    self._log("error", "command missing 'type'")
                    continue
                with self._stdin_lock:
                    self._cmd_queue.append(obj)
                if obj.get("type") == "shutdown":
                    self._stop_flag.set()
                    return
        except Exception as e:  # pragma: no cover
            self._log("error", f"stdin reader died: {e}")
            self._stop_flag.set()

    def _drain_commands(self) -> None:
        with self._stdin_lock:
            cmds = self._cmd_queue[:]
            self._cmd_queue.clear()

        # Coalesce bursts of `apply_state` commands (e.g. XY-pad drag at 20 Hz):
        # only the most-recent state matters by the time we apply it. Earlier
        # ones are obsolete the instant a newer one arrives, so dispatching all
        # of them just wastes time on the audio thread and risks underruns.
        # We still preserve any seed-change instruction from earlier commands
        # so a deliberate seed crossfade isn't dropped silently.
        coalesced: list = []
        pending_apply: Optional[Dict[str, Any]] = None
        for cmd in cmds:
            if cmd.get("type") == "apply_state":
                # If an older pending apply_state carried a seed and the new
                # one doesn't, propagate it forward so the user's intent is
                # preserved.
                if (pending_apply is not None
                        and pending_apply.get("seed") is not None
                        and cmd.get("seed") is None):
                    cmd = dict(cmd)
                    cmd["seed"] = pending_apply["seed"]
                    if pending_apply.get("seed_mode") is not None:
                        cmd["seed_mode"] = pending_apply["seed_mode"]
                pending_apply = cmd
            else:
                if pending_apply is not None:
                    coalesced.append(pending_apply)
                    pending_apply = None
                coalesced.append(cmd)
        if pending_apply is not None:
            coalesced.append(pending_apply)

        for cmd in coalesced:
            try:
                self._dispatch(cmd)
            except Exception as e:
                self._log("error", f"command {cmd.get('type','?')} failed: {e}")

    # ----- command dispatch ----- #

    def _dispatch(self, cmd: Dict[str, Any]) -> None:
        t = cmd.get("type")
        gen = self.gen
        if t == "shutdown":
            self._stop_flag.set()
            return
        if t == "set_param":
            name = cmd["name"]
            val = cmd["value"]
            if name == "temperature":
                self._base_temperature = float(val)
            elif name == "n_steps":
                self._base_n_steps = max(1, int(val))
            elif name == "solver":
                solver = str(val).lower()
                if solver in ("euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong"):
                    gen.solver = solver  # not LFO'd
            elif name == "schedule":
                sched = str(val).lower()
                if sched in ("linear", "shifted"):
                    gen.schedule = sched
            elif name == "schedule_shift":
                gen.schedule_shift = float(val)
            elif name == "context_chunks":
                self._base_context_chunks = max(0, min(int(val), gen.max_context_chunks))
                gen.set_context_chunks(self._base_context_chunks)
            elif name == "crossfade_chunks":
                self._base_crossfade_chunks = max(1, min(16, int(val)))
                gen.crossfade_chunks = self._base_crossfade_chunks
            elif name == "auto_cycle_enabled":
                gen.auto_cycle_enabled = bool(val)
            elif name == "auto_cycle_interval":
                gen.auto_cycle_interval = max(1, int(val))
            elif name == "master_gain":
                # Output volume (0 .. 2). Applied in the audio callback.
                self._master_gain = float(max(0.0, min(4.0, float(val))))
            elif name == "input_gain":
                # Input (mic) gain, applied to mono samples before encoding.
                self.input.input_gain = float(max(0.0, min(8.0, float(val))))
            elif name == "input_mode_enabled":
                self.input.enabled = bool(val)
            elif name == "input_blend":
                # 0 = ignore mic, 1 = replace last block of prefix with mic.
                self.input.blend = float(max(0.0, min(1.0, float(val))))
            else:
                self._log("error", f"unknown param: {name}")
            return
        if t == "audio_input":
            # Frontend pushes a chunk of mono mic samples here. Wire format:
            #   { "type":"audio_input", "sr":48000, "b64":"<float32 LE base64>" }
            #   or
            #   { "type":"audio_input", "sr":48000, "samples":[float, ...] }
            try:
                src_sr = int(cmd.get("sr") or self.gen.sample_rate)
                if "b64" in cmd:
                    raw = base64.b64decode(cmd["b64"])
                    mono = np.frombuffer(raw, dtype=np.float32)
                else:
                    mono = np.asarray(cmd.get("samples") or [], dtype=np.float32)
                if mono.size:
                    self.input.push_samples(mono, src_sr)
            except Exception as e:
                self._log("error", f"audio_input failed: {e}")
            return
        if t == "set_summary_token":
            idx = int(cmd["idx"])
            kind = str(cmd["kind"]).lower()
            value = float(cmd["value"])
            if kind in ("s", "scale"):
                self._base_summary_scale[idx] = value
            elif kind in ("b", "bias"):
                self._base_summary_bias[idx] = value
            else:
                self._log("error", f"set_summary_token bad kind: {kind}")
            return
        if t == "set_channel_dim":
            idx = int(cmd["idx"])
            kind = str(cmd["kind"]).lower()
            value = float(cmd["value"])
            if kind in ("s", "scale"):
                self._base_channel_scale[idx] = value
            elif kind in ("b", "bias"):
                self._base_channel_bias[idx] = value
            else:
                self._log("error", f"set_channel_dim bad kind: {kind}")
            return
        if t == "set_summary_vector":
            kind = str(cmd["kind"]).lower()
            vals = torch.as_tensor(cmd["values"], dtype=torch.float32, device=gen.device)
            if vals.numel() != gen.block_size:
                self._log("error", f"summary vector length must be {gen.block_size}")
                return
            if kind in ("s", "scale"):
                self._base_summary_scale = vals.clone()
            else:
                self._base_summary_bias = vals.clone()
            return
        if t == "set_channel_vector":
            kind = str(cmd["kind"]).lower()
            vals = torch.as_tensor(cmd["values"], dtype=torch.float32, device=gen.device)
            if vals.numel() != gen.latent_dim:
                self._log("error", f"channel vector length must be {gen.latent_dim}")
                return
            if kind in ("s", "scale"):
                self._base_channel_scale = vals.clone()
            else:
                self._base_channel_bias = vals.clone()
            return
        if t == "randomize":
            group = str(cmd.get("group", "summary")).lower()
            kind = str(cmd.get("kind", "both")).lower()
            if group == "summary":
                if kind in ("scale", "both"):
                    gen.randomize_summary_scale()
                    self._base_summary_scale = gen.summary_scale.detach().clone()
                if kind in ("bias", "both"):
                    gen.randomize_summary_bias()
                    self._base_summary_bias = gen.summary_bias.detach().clone()
            elif group == "channel":
                if kind in ("scale", "both"):
                    gen.randomize_channel_scale()
                    self._base_channel_scale = gen.channel_scale.detach().clone()
                if kind in ("bias", "both"):
                    gen.randomize_channel_bias()
                    self._base_channel_bias = gen.channel_bias.detach().clone()
            else:
                self._log("error", f"randomize: bad group {group}")
            return
        if t == "reset":
            group = str(cmd.get("group", "all")).lower()
            if group in ("summary", "all"):
                gen.reset_summary_control()
                self._base_summary_scale = gen.summary_scale.detach().clone()
                self._base_summary_bias = gen.summary_bias.detach().clone()
            if group in ("channel", "all"):
                gen.reset_channel_control()
                self._base_channel_scale = gen.channel_scale.detach().clone()
                self._base_channel_bias = gen.channel_bias.detach().clone()
            if group == "all":
                # reset scalars too
                self._base_temperature = 1.0
                self._base_n_steps = 4
                self._base_context_chunks = gen.default_context_chunks
                self._base_crossfade_chunks = 4
                gen.solver = "euler"
                gen.set_context_chunks(self._base_context_chunks)
                gen.crossfade_chunks = self._base_crossfade_chunks
                for lfo in self.lfos.values():
                    lfo.enabled = False
            return
        if t == "new_seed":
            mode = str(cmd.get("mode", "crossfade")).lower()
            if "seed" in cmd and cmd["seed"] is not None:
                seed = int(cmd["seed"]) % (2 ** 31)
            else:
                seed = random.randint(0, 2 ** 31 - 1)
            if mode == "hardcut":
                gen.set_seed(seed)
            else:
                gen.crossfade_to_seed(seed)
            return
        if t == "set_lfo":
            name = cmd.get("name")
            if name not in self.lfos:
                self._log("error", f"unknown LFO: {name}")
                return
            self.lfos[name].update(cmd)
            return
        if t == "apply_state":
            # Atomic preset/morph apply: scalars + 8/64-dim vectors + optional seed.
            # Used by the Electron preset XY-pad to stream interpolated states.
            p = cmd.get("params") or {}
            if "temperature" in p:
                self._base_temperature = float(p["temperature"])
            if "n_steps" in p:
                self._base_n_steps = max(1, int(p["n_steps"]))
            if "solver" in p:
                solver = str(p["solver"]).lower()
                if solver in ("euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong"):
                    gen.solver = solver
            if "context_chunks" in p:
                v = max(0, min(int(p["context_chunks"]), gen.max_context_chunks))
                self._base_context_chunks = v
                gen.set_context_chunks(v)
            if "crossfade_chunks" in p:
                self._base_crossfade_chunks = max(1, min(16, int(p["crossfade_chunks"])))
                gen.crossfade_chunks = self._base_crossfade_chunks
            # Build the vector tensors via numpy first (fast list->ndarray),
            # then a single CPU->device transfer. This is noticeably cheaper
            # than `torch.as_tensor(list, device='mps')` for short vectors,
            # and avoids the unnecessary `.clone()` (as_tensor already
            # produces a fresh tensor when its source is a Python list).
            for key, n in (("summary_scale", gen.block_size),
                           ("summary_bias", gen.block_size),
                           ("channel_scale", gen.latent_dim),
                           ("channel_bias", gen.latent_dim)):
                arr = cmd.get(key)
                if arr is None:
                    continue
                if not hasattr(arr, "__len__") or len(arr) != n:
                    self._log("error", f"apply_state: {key} must have length {n}")
                    continue
                np_arr = np.asarray(arr, dtype=np.float32)
                tensor = torch.from_numpy(np_arr).to(gen.device, non_blocking=True)
                if key == "summary_scale":
                    self._base_summary_scale = tensor
                elif key == "summary_bias":
                    self._base_summary_bias = tensor
                elif key == "channel_scale":
                    self._base_channel_scale = tensor
                elif key == "channel_bias":
                    self._base_channel_bias = tensor
            if cmd.get("seed") is not None:
                seed = int(cmd["seed"]) % (2 ** 31)
                mode = str(cmd.get("seed_mode", "crossfade")).lower()
                if mode == "hardcut":
                    gen.set_seed(seed)
                else:
                    gen.crossfade_to_seed(seed)
            return
        self._log("error", f"unknown command: {t}")

    # ----- waveform pusher (background thread, ~30 Hz) ----- #

    def _waveform_pusher_loop(self) -> None:
        """Push a downsampled waveform of *recently played* audio at ~20 Hz.

        Designed to never interfere with the audio callback:
          - Small visualisation window (~0.5 s) so the lock-protected copy
            inside `recent_waveform()` is sub-millisecond.
          - Best-effort cadence with explicit `time.sleep()` so the thread
            never spins; releases the GIL so the audio callback / chunk
            generator are not starved of Python time.
          - Wrapped in broad try/except: a viz failure can never stop audio.
        """
        gen = self.gen
        target_points = 256
        # 0.5 s of recently-played audio — short enough to copy in <0.1 ms,
        # long enough to look like a meaningful oscilloscope trace.
        window_samples = int(0.5 * gen.sample_rate)
        bin_size = max(1, window_samples // target_points)
        period = 1.0 / 20.0  # 20 Hz refresh; visually smooth at 60 fps render
        try:
            while not self._stop_flag.is_set():
                start = time.perf_counter()
                try:
                    w = gen.audio_buffer.recent_waveform(window_samples)
                except Exception:
                    w = None
                if w is not None and w.size:
                    n = (w.size // bin_size) * bin_size
                    if n > 0:
                        view = w[:n].reshape(-1, bin_size)
                        amax = view.max(axis=1)
                        amin = view.min(axis=1)
                        peaks = np.where(np.abs(amax) >= np.abs(amin), amax, amin)
                        samples = peaks[-target_points:].astype(np.float32).tolist()
                        self._emit({"type": "waveform", "samples": samples})
                elapsed = time.perf_counter() - start
                sleep_for = period - elapsed
                # Always yield at least a tiny slice so the GIL is released.
                time.sleep(max(sleep_for, 0.001))
        except Exception as e:
            try:
                self._log("warn", f"waveform pusher exited: {e}")
            except Exception:
                pass

    # ----- LFO application ----- #

    def _apply_modulation(self) -> Dict[str, Any]:
        """Resolve effective values from base + LFO, push them into the generator,
        and return a snapshot used in the next chunk event."""
        gen = self.gen
        t_s = time.time() - self._t_start
        L = self.lfos

        # Scalars: temperature, n_steps, context_chunks, crossfade_chunks
        gen.temperature = max(0.05, self._base_temperature + L["temperature"].value(t_s))
        live_steps = self._base_n_steps + int(round(L["n_steps"].value(t_s)))
        gen.n_steps = int(max(1, min(64, live_steps)))
        live_ctx = self._base_context_chunks + int(round(L["context_chunks"].value(t_s)))
        live_ctx = max(0, min(gen.max_context_chunks, live_ctx))
        if live_ctx != gen.context_chunks:
            gen.set_context_chunks(live_ctx)
        live_xf = self._base_crossfade_chunks + int(round(L["crossfade_chunks"].value(t_s)))
        gen.crossfade_chunks = int(max(1, min(16, live_xf)))

        # Vectors: add a uniform LFO scalar to every element of the static base.
        ss_off = L["summary_scale"].value(t_s)
        sb_off = L["summary_bias"].value(t_s)
        cs_off = L["channel_scale"].value(t_s)
        cb_off = L["channel_bias"].value(t_s)
        gen.summary_scale = (self._base_summary_scale + ss_off).clamp_(0.01, 8.0)
        gen.summary_bias = (self._base_summary_bias + sb_off).clamp_(-8.0, 8.0)
        gen.channel_scale = (self._base_channel_scale + cs_off).clamp_(0.01, 8.0)
        gen.channel_bias = (self._base_channel_bias + cb_off).clamp_(-8.0, 8.0)

        return {
            "temperature": float(gen.temperature),
            "n_steps": int(gen.n_steps),
            "context_chunks": int(gen.context_chunks),
            "crossfade_chunks": int(gen.crossfade_chunks),
            "lfo_offsets": {
                "temperature": float(L["temperature"].value(t_s)),
                "n_steps": float(L["n_steps"].value(t_s)),
                "context_chunks": float(L["context_chunks"].value(t_s)),
                "crossfade_chunks": float(L["crossfade_chunks"].value(t_s)),
                "summary_scale": float(ss_off),
                "summary_bias": float(sb_off),
                "channel_scale": float(cs_off),
                "channel_bias": float(cb_off),
            },
        }

    # ----- snapshot for ready event ----- #

    def _snapshot(self) -> Dict[str, Any]:
        gen = self.gen
        return {
            "block_size": int(gen.block_size),
            "latent_dim": int(gen.latent_dim),
            "sample_rate": int(gen.sample_rate),
            "samples_per_chunk": int(gen.samples_per_chunk),
            "max_context_chunks": int(gen.max_context_chunks),
            "default_context_chunks": int(gen.default_context_chunks),
            "summary_scale_range": list(gen.summary_scale_range),
            "summary_bias_range": list(gen.summary_bias_range),
            "channel_scale_range": list(gen.channel_scale_range),
            "channel_bias_range": list(gen.channel_bias_range),
            "lfo_shapes": list(LFO_SHAPES),
            "lfo_names": list(LFO_NAMES),
            "params": {
                "temperature": self._base_temperature,
                "n_steps": self._base_n_steps,
                "solver": gen.solver,
                "schedule": getattr(gen, "schedule", "linear"),
                "schedule_shift": float(getattr(gen, "schedule_shift", 0.0)),
                "context_chunks": self._base_context_chunks,
                "crossfade_chunks": self._base_crossfade_chunks,
                "seed": int(gen.current_seed) if gen.current_seed is not None else None,
                "master_gain": float(self._master_gain),
                "input_gain": float(self.input.input_gain),
                "input_mode_enabled": bool(self.input.enabled),
                "input_blend": float(self.input.blend),
            },
            "summary_scale": self._base_summary_scale.detach().cpu().tolist(),
            "summary_bias": self._base_summary_bias.detach().cpu().tolist(),
            "channel_scale": self._base_channel_scale.detach().cpu().tolist(),
            "channel_bias": self._base_channel_bias.detach().cpu().tolist(),
            "lfos": {n: l.to_dict() for n, l in self.lfos.items()},
            "output_devices": _list_output_devices(),
            "output_device_index": (
                int(self._output_device) if self._output_device is not None else None
            ),
        }

    # ----- main loop ----- #

    def run(self) -> None:
        gen = self.gen
        gen.model.eval()
        gen.codec.reset_streaming()
        gen._open_save_writer(self.save_path)

        # ready event
        self._emit({"type": "ready", **self._snapshot()})

        # Pre-buffer (no audio output yet)
        for i in range(gen.prebuffer_chunks):
            self._apply_modulation()
            audio, t_gen, t_dec = gen._step()
            gen.audio_buffer.push(audio)
            gen._save_chunk(audio)
            self._emit_chunk_event(audio_len=len(audio), t_gen=t_gen, t_dec=t_dec,
                                   prebuffer=True, idx=i)

        # Audio callback pulls from generator's ring buffer and applies the
        # master output gain. Reading `self._master_gain` is atomic in CPython
        # for floats (single bytecode), so no lock is needed.
        def _callback(outdata, frames, time_info, status):
            buf = gen.audio_buffer.pull(frames)
            g = self._master_gain
            if g != 1.0:
                np.multiply(buf, g, out=buf)
            outdata[:] = buf

        chunks_done = gen.prebuffer_chunks
        # Buffer window: large enough to absorb any single ~2 s compute spike
        # (snap-load `crossfade_to_seed`, occasional GC pause, MPS cache
        # release, etc.) without ever underrunning, while still 6× tighter
        # than the original 30 s — i.e. live param / preset changes are heard
        # within ~3-5 s, never with audio dropouts.
        max_buf_secs = 5.0
        throttle_to = 3.0

        # Start the background waveform pusher (push model, no polling).
        wf_thread = threading.Thread(target=self._waveform_pusher_loop, daemon=True)
        wf_thread.start()

        def _open_output_stream():
            """Open the audio output stream, falling back to system default if
            the configured device fails (e.g. wrong channel count, samplerate,
            unplugged interface). Returns the open OutputStream context."""
            try:
                return sd.OutputStream(
                    samplerate=gen.sample_rate,
                    channels=2,
                    dtype="float32",
                    blocksize=2048,
                    callback=_callback,
                    latency="low",
                    device=self._output_device,
                )
            except Exception as e:
                if self._output_device is None:
                    raise
                sys.stderr.write(
                    f"[server] failed to open output device {self._output_device!r}: {e}; "
                    f"falling back to system default\n"
                )
                self._output_device = None
                return sd.OutputStream(
                    samplerate=gen.sample_rate,
                    channels=2,
                    dtype="float32",
                    blocksize=2048,
                    callback=_callback,
                    latency="low",
                    device=None,
                )

        try:
            with _open_output_stream():
                while not self._stop_flag.is_set():
                    self._drain_commands()

                    if (self.max_chunks is not None
                            and chunks_done >= self.max_chunks):
                        while gen.audio_buffer.buffered_samples > 0:
                            time.sleep(0.05)
                        time.sleep(0.3)
                        break

                    # Throttle generation if buffer is too deep.
                    buf_s = gen.audio_buffer.buffered_samples / gen.sample_rate
                    if buf_s > max_buf_secs:
                        while (gen.audio_buffer.buffered_samples / gen.sample_rate
                               > throttle_to and not self._stop_flag.is_set()):
                            self._drain_commands()
                            time.sleep(0.05)
                        continue

                    mod_snapshot = self._apply_modulation()
                    # If the user has live mic input enabled, inject the
                    # most recent mic-encoded latents into `gen.prefix_norm`
                    # so the next chunk is conditioned on what they're
                    # playing. This is a no-op fast path when input is off.
                    self.input.apply_to_prefix()
                    try:
                        audio, t_gen, t_dec = gen._step()
                        gen._consecutive_errors = 0
                    except Exception as e:
                        gen._consecutive_errors += 1
                        self._log("error",
                                  f"step {chunks_done+1} failed: {e} "
                                  f"(consecutive={gen._consecutive_errors})")
                        if gen._consecutive_errors >= 5:
                            self._log("error", "5 consecutive failures, aborting")
                            break
                        gen._full_reset()
                        gen._set_time_seed()
                        time.sleep(0.05)
                        continue

                    gen.audio_buffer.push(audio)
                    gen._save_chunk(audio)
                    chunks_done += 1
                    gen._maybe_auto_cycle()

                    self._emit_chunk_event(audio_len=len(audio),
                                           t_gen=t_gen, t_dec=t_dec,
                                           idx=chunks_done,
                                           mod=mod_snapshot,
                                           buf_s=buf_s)
        finally:
            gen._close_save_writer()
            self._emit({"type": "stopped"})

    def _emit_chunk_event(self, *, audio_len: int, t_gen: float, t_dec: float,
                          idx: int, prebuffer: bool = False,
                          mod: Optional[Dict[str, Any]] = None,
                          buf_s: Optional[float] = None) -> None:
        gen = self.gen
        chunk_secs = audio_len / gen.sample_rate
        total_t = t_gen + t_dec
        rtf = total_t / chunk_secs if chunk_secs > 0 else float("inf")

        # Cheap audio peak (for a meter). Skip true RMS to keep cost trivial.
        # audio_buffer holds float32 stereo arrays already; we stat the just-
        # pushed chunk via gen.audio_buffer's last buffered samples is more
        # work, so we recompute on the chunk we just pushed.
        # Note: audio is float32 [-1,1].
        try:
            # We don't have a handle to `audio` here; skip and let frontend
            # estimate from latent stats if needed. Could be added later.
            peak = None
        except Exception:
            peak = None

        # Latent stats for the most recent chunk: per-token mean over channels
        # and per-channel mean over tokens, both from the *base* (post-LFO)
        # values that were just written. Keeps payload small (~580 floats).
        ss = gen.summary_scale.detach().cpu().tolist()
        sb = gen.summary_bias.detach().cpu().tolist()
        cs = gen.channel_scale.detach().cpu().tolist()
        cb = gen.channel_bias.detach().cpu().tolist()

        # Recent chunk's latent (in normalized space) is the last block_size
        # tokens of the prefix (we just appended it in _step()). Provide the
        # full [block_size, latent_dim] grid so the front-end can paint the
        # mandala from real model output.
        try:
            tail = gen.prefix_norm[:, -gen.block_size:, :].detach().cpu()
            latent_grid = tail.squeeze(0).tolist()  # [block_size][latent_dim]
        except Exception:
            latent_grid = None

        evt: Dict[str, Any] = {
            "type": "chunk",
            "n": int(idx),
            "prebuffer": bool(prebuffer),
            "gen_ms": round(t_gen * 1000.0, 1),
            "dec_ms": round(t_dec * 1000.0, 1),
            "rtf": round(rtf, 3),
            "buf_s": round(buf_s if buf_s is not None
                           else gen.audio_buffer.buffered_samples / gen.sample_rate, 2),
            "underruns": int(gen.audio_buffer.underruns),
            "seed": int(gen.current_seed) if gen.current_seed is not None else None,
            "summary_scale": ss,
            "summary_bias": sb,
            "channel_scale_stats": _vec_stats(cs),
            "channel_bias_stats": _vec_stats(cb),
            # Send full channel vectors only every 4 chunks to keep payload small.
            "channel_scale": cs if (idx % 4 == 0 or prebuffer) else None,
            "channel_bias": cb if (idx % 4 == 0 or prebuffer) else None,
            "latent_grid": latent_grid,  # last chunk in normalized space
            "params": {
                "temperature": float(gen.temperature),
                "n_steps": int(gen.n_steps),
                "solver": gen.solver,
                "schedule": getattr(gen, "schedule", "linear"),
                "schedule_shift": float(getattr(gen, "schedule_shift", 0.0)),
                "context_chunks": int(gen.context_chunks),
                "crossfade_chunks": int(gen.crossfade_chunks),
            },
        }
        if mod is not None:
            evt["mod"] = mod
        self._emit(evt)


def _vec_stats(v):
    if not v:
        return None
    a = np.asarray(v, dtype=np.float32)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Realtime FlowDiT generation server (stdio JSON IPC)."
    )
    p.add_argument("--ckpt", required=True, help="Path to last.pt or ema.pt.")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--device", default=None, help="cuda | mps | cpu")
    p.add_argument("--output-device", default=None,
                   help="sounddevice output device (int index or name substring). "
                        "Defaults to the system default output.")
    p.add_argument("--nfe", type=int, default=4, dest="n_steps")
    p.add_argument("--solver", default="euler", choices=["euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed-scale", type=float, default=0.0)
    p.add_argument("--context-chunks", type=int, default=32)
    p.add_argument("--prebuffer", type=int, default=2)
    p.add_argument("--crossfade-chunks", type=int, default=4)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--coreml-path", default=None,
                   help="Path to CoreML model (.mlpackage) for inference. "
                        "If provided, uses CoreML backend with fallback to PyTorch.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    if sd is None:
        sys.stderr.write("sounddevice is required.\n")
        sys.exit(2)

    device = best_device(args.device)
    ckpt_path = str(Path(args.ckpt).expanduser().resolve())
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    model = load_model(ckpt_path, device, use_ema=bool(args.use_ema))
    codec = CodecWrapper(CodecConfig(device=str(device)))

    gen = RealtimeFlowGenerator(
        model=model,
        codec=codec,
        device=device,
        n_steps=args.n_steps,
        solver=args.solver,
        temperature=args.temperature,
        seed_scale=args.seed_scale,
        context_chunks=args.context_chunks,
        prebuffer_chunks=args.prebuffer,
        crossfade_chunks=args.crossfade_chunks,
        initial_seed=args.seed,
        coreml_path=args.coreml_path,
    )

    app = ServerApp(gen, max_chunks=args.max_chunks, save_path=args.save,
                    output_device=args.output_device)
    try:
        app.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
