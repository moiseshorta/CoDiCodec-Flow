# CoDiCodec-Flow: Realtime audio generation model using Flow Matching DiT on CoDiCodec latents.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moiseshorta/CoDiCodec-Flow/blob/main/colab/CoDiCodec_Flow.ipynb)

> Author: Moisés Horta Valenzuela, [hexorcismos](http://hexorcismos.bandcamp.com)
> 
> Date: May 2026

A generative model that synthesises audio in CoDiCodec's continuous latent space
using Conditional Flow Matching (CFM) on a block-causal DiT architecture.

The model targets **musical continuation / improvising accompaniment**: given a
short audio prompt, it generates an arbitrarily long continuation in a
chunk-causal, streaming fashion on the codec's ~11.7 Hz, 64-channel latent
sequence.

## Design

- **CoDiCodec** ([Pasini et al., 2025](https://arxiv.org/pdf/2509.09836))
  encodes 48 kHz stereo audio to summary embeddings at ~11.7 Hz with 64
  channels (128x compression) and exposes a streaming `decode_next()` API.
  The continuous latents, after the codec's `atanh / sigma_rescale=0.8`
  transform, are approximately unit-Gaussian — a direct fit for flow
  matching.
- **Block-causal Flow Matching DiT** is the simplest architecture that:
  1. respects the codec's chunk structure,
  2. supports KV-caching for efficient streaming inference,
  3. has unconditional dropout-based classifier-free guidance for free.
- The whole pipeline is **MPS-friendly** so it can be trained and run in
  real-time on a 36 GB Apple Silicon laptop.

## Google Colab

Try CoDiCodec-Flow directly in your browser using the Google Colab notebook in the `colab/` directory. The notebook provides a step-by-step guide for:

- Cloning the repository and setting up the environment
- Preprocessing audio data to latents
- Training a model on your data
- Generating audio continuations

Click the badge at the top of this README to open the notebook in Colab.

## Repository layout

```
codicodec-flow/
  codicodec/           upstream codec package (do not modify)
  flow/                this project
    __init__.py
    codec_wrapper.py   MPS-safe wrapper around codicodec.EncoderDecoder
    config.py          dataclass-based config
    utils.py           device, masks, logging
    data/
      preencode.py     audio dir -> per-file .pt latent shards
      latent_dataset.py
    model/
      dit.py           block-causal flow-matching DiT
      cfm.py           CFM loss + Euler/Heun samplers
      ema.py
    train.py           training loop
    sample.py          offline sampling
    smoke_test.py      end-to-end sanity check
  requirements.txt
  README.md            (this file)
```

## Installation

```bash
# Create conda environment
conda create -n codicodec-flow python=3.10
conda activate codicodec-flow

# Install dependencies
pip install -r requirements.txt

# Install the upstream CoDiCodec package
pip install -e ./codicodec

# Verify codec works on your machine (downloads checkpoint on first run)
python -m flow.smoke_test --device mps  # Use 'cuda' for NVIDIA GPUs
```

## CLI Interface

codicodec-flow provides a user-friendly CLI wrapper that simplifies training, preprocessing, and generation without requiring `python -m flow...` commands.

```bash
# Preprocess audio data
python cli.py preprocess --in-dir ~/music/training --out-dir ./data/latents --device mps

# Train a model (TUI monitoring enabled by default)
python cli.py train --data-dir ./data/latents --out-dir ./runs/v0 --device mps

# Generate audio
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --out ./out.wav --device mps

# Stream audio indefinitely with live keyboard controls (press 'q' to quit)
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema --device mps
```

See [Real-time Streaming Generation](#real-time-streaming-generation) for the full list of options and live keyboard controls.

## Preprocessing

Before training, you need to convert your audio files into latent shards using the CoDiCodec encoder.

```bash
python cli.py preprocess \
    --in-dir   /path/to/your/audio \
    --out-dir  ./data/latents      \
    --device   mps                 \
    --max-seconds 60
```

**Arguments:**
- `--in-dir`: Directory containing audio files (WAV, MP3, FLAC, etc.)
- `--out-dir`: Output directory for latent shards (.pt files)
- `--device`: `mps` for Apple Silicon, `cuda` for NVIDIA GPUs, `cpu` as fallback
- `--max-seconds`: Maximum duration per file (default: 300s). Longer files are split.

**Output:**
- Each audio file produces a `.pt` file containing the encoded latent representation
- Latents are stored as `[T, 8, 64]` tensors (T = number of 0.683s chunks)
- Files are stored with metadata for the dataset loader

**Tips:**
- Use diverse audio for better generalization (different styles, instruments, tempos)
- 48 kHz stereo audio is recommended (CoDiCodec's native rate)
- Aim for several hours of audio for reasonable training
- Train for at least 100K steps for meaningful results; the v3_okachihuali model was trained for 6,860,000 steps

## Training

Train a block-causal Flow Matching DiT model on the preprocessed latents.

```bash
python cli.py train \
    --data-dir   ./data/latents \
    --out-dir    ./runs/v0      \
    --device     mps            \
    --batch-size 4              \
    --grad-accum 2              \
    --crop-tokens 512          \
    --max-steps 200000
```

**Key Arguments:**
- `--data-dir`: Directory containing preprocessed latent shards
- `--out-dir`: Output directory for checkpoints and logs
- `--device`: `mps`, `cuda`, or `cpu`
- `--batch-size`: Batch size per GPU (default: 8, use 4 on MPS)
- `--grad-accum`: Gradient accumulation steps (effective batch = batch_size × grad_accum)
- `--crop-tokens`: Random crop length in tokens (default: 768, must be multiple of 8)
- `--max-steps`: Total training steps (default: 200000)
- `--dtype`: `bf16` for bfloat16 (faster, less memory) or `fp32` for float32
- `--lr`: Learning rate (default: 1e-4 with cosine decay)
- `--ema-decay`: EMA decay rate (default: 0.9999)

**Model Size Configuration:**

Default (~97M params, recommended for 36GB+ RAM):
```bash
python cli.py train --data-dir ./data/latents --out-dir ./runs/v0 \
    --device mps --batch-size 4 --grad-accum 2 --crop-tokens 512 \
    --dtype bf16 --max-steps 200000
```

Smaller (~20M params, faster iteration):
```bash
python cli.py train --data-dir ./data/latents --out-dir ./runs/v0 \
    --device mps --batch-size 8 --grad-accum 2 --crop-tokens 512 \
    --dtype bf16 --max-steps 200000 \
    --dim 384 --n-layers 8 --n-heads 6 --cond-dim 384
```

**Training Details:**
- Checkpoints are saved every 50 steps: `last.pt` (latest) and `ema.pt` (EMA copy)
- Periodic audio samples are generated during training (unconditional by default)
- Use `--audio-continuation` to enable continuation sampling during training
- Use `--audio-sample-every N` to control sampling frequency (0 to disable)
- Logs include loss, learning rate, and sample metrics

## Generation

Generate audio continuations using a trained checkpoint.

### Continuation from a prompt

```bash
python cli.py sample \
    --ckpt        ./runs/v0/ema.pt \
    --prompt-wav  ./prompt.wav     \
    --duration-s  20               \
    --nfe         8                \
    --solver      heun             \
    --out         ./out.wav        \
    --device      mps
```

**Arguments:**
- `--ckpt`: Path to checkpoint (use `ema.pt` for best quality, `last.pt` for latest)
- `--prompt-wav`: Audio prompt file (WAV, 48 kHz stereo recommended)
- `--duration-s`: Duration of continuation in seconds (default: 20)
- `--nfe`: Number of function evaluations (sampling steps, default: 8)
- `--solver`: Sampler (default `heun`). Supported:
  - `euler`    : 1st-order ODE, 1 NFE/step (fastest baseline)
  - `heun`     : 2nd-order ODE, 2 NFE/step (default)
  - `midpoint` : 2nd-order RK2, 2 NFE/step
  - `rk4`      : 4th-order Runge-Kutta, 4 NFE/step (highest quality at low NFE)
  - `dpmpp`    : DPM-Solver++ 2M for rectified flow, 1 NFE/step (strong at NFE 4-8)
  - `pingpong` : stochastic SDE sampler, 1 NFE/step (recommended for distilled / few-step models)
- `--schedule`: Time grid: `linear` (default) or `shifted` (logSNR shift)
- `--schedule-shift`: LogSNR shift exponent for `--schedule shifted` (e.g. `1.0`-`3.0` for long sequences)
- `--out`: Output audio file path
- `--device`: `mps`, `cuda`, or `cpu`
- `--temperature`: Sampling temperature (default: 1.0, higher = more diverse)
- `--n-steps`: Number of diffusion steps (default: 32)

### Unconditional generation

```bash
python cli.py sample \
    --ckpt        ./runs/v0/ema.pt \
    --duration-s  20               \
    --nfe         8                \
    --solver      heun             \
    --out         ./out_uncond.wav \
    --device      mps
```

Omit `--prompt-wav` for unconditional generation (no prompt context).

### Advanced options

**Higher quality with more sampling steps:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 30 --nfe 16 --solver heun --out ./out_high_quality.wav --device mps
```

**Faster generation with fewer steps:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 4 --solver euler --out ./out_fast.wav --device mps
```

**Adjust temperature for diversity:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 8 --solver heun --temperature 1.5 --out ./out_diverse.wav --device mps
```

**DPM-Solver++ for very low NFE:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 6 --solver dpmpp --out ./out_dpmpp.wav --device mps
```

**RK4 for highest quality at low NFE:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 4 --solver rk4 --out ./out_rk4.wav --device mps
```

**Sampling Trade-offs:**
- **NFE (steps)**: More steps = better quality but slower. 4-8 is real-time, 16+ is high quality.
- **Solver**: `euler`/`dpmpp`/`pingpong` use 1 NFE per step; `heun`/`midpoint` use 2; `rk4` uses 4. `dpmpp` and `rk4` give the best quality at very low total NFE; `pingpong` adds stochasticity (only recommended for distilled models).
- **Schedule**: `--schedule shifted --schedule-shift 1.0` warps the time grid in log-SNR space (SD3-style); helpful for long sequences or low NFE.
- **Temperature**: Higher values increase diversity but may reduce coherence.

## Real-time Streaming Generation

The `realtime` subcommand spins up a streaming generator that **plays audio
indefinitely** out of the system's default sound device while the FlowDiT
model continually fills a sliding-window buffer in the background. It is the
same engine used by the Electron GUI, exposed as a standalone interactive
terminal app. Use it for headless live-coding sessions, long-form ambient
playback, or stress-testing a checkpoint.

### Quick start

```bash
# Minimal: stream from a checkpoint until you press 'q'
python cli.py realtime --ckpt ./runs/v3_okachihuali/ema.pt --use-ema --device mps

# Same thing, but called directly as a module
python -m flow.realtime --ckpt ./runs/v3_okachihuali/ema.pt --use-ema --device mps
```

On launch you'll see the engine load the model, prebuffer a couple of chunks
(~1.4 s @ 48 kHz), and then start playback. The terminal stays in raw mode so
single keypresses act as live controls — there is no need to press Enter.

### Common recipes

```bash
# High-quality, slightly slower (8 ODE steps with Heun integrator)
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema \
    --solver heun --nfe 8 --temperature 0.95

# Fast, low-NFE setup (good for live performance on a laptop)
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema \
    --solver dpmpp --nfe 4

# Long context window for more coherent long-form output
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema \
    --context-chunks 64 --prebuffer 4 --crossfade-chunks 8

# Capture the whole session to a WAV file while playing it live
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema \
    --save ./session.wav

# Reproducible session: fix the initial seed
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema --seed 12345

# Stop automatically after 200 chunks (~137 s @ 48 kHz)
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema --max-chunks 200

# Use a converted CoreML model (auto-falls back to PyTorch on shape mismatch)
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema \
    --coreml-path ./runs/v0/model.mlpackage
```

### CLI options

| Flag | Default | Description |
| --- | --- | --- |
| `--ckpt PATH` | *required* | Checkpoint to load (`last.pt` or `ema.pt`). |
| `--use-ema` | off | Load EMA weights — strongly recommended for inference. |
| `--device DEV` | auto | `mps`, `cuda`, or `cpu`. Auto-detected if omitted. |
| `--coreml-path PATH` | none | Optional CoreML `.mlpackage`. Falls back to PyTorch when the variable sliding-window shape does not match the traced shape. |
| `--nfe N` | `4` | ODE steps per chunk. Higher = better quality, lower = lower latency. |
| `--solver NAME` | `euler` | One of `euler`, `heun`, `midpoint`, `rk4`, `dpmpp`, `pingpong`. |
| `--temperature F` | `1.0` | Velocity scaling. `<1` sharpens, `>1` diffuses. |
| `--seed-scale F` | `0.0` | Shrinks initial noise toward zero (0 = standard `N(0, I)`). |
| `--context-chunks N` | `32` | Sliding-window length in codec chunks (1 chunk ≈ 0.683 s @ 48 kHz). Pass `<= 0` to use the model's maximum safe context. |
| `--prebuffer N` | `2` | Chunks to render before playback starts. Higher = more glitch-resistant, more startup latency. |
| `--crossfade-chunks N` | `4` | Crossfade length used when switching seeds mid-stream. |
| `--max-chunks N` | unbounded | Stop after `N` chunks. Omit for indefinite playback. |
| `--save PATH` | none | Write the full session to a `.wav` file alongside live playback. |
| `--seed N` | time-based | Initial RNG seed. |
| `--summary-scale V` | `1.0` | Initial summary-latent scale. Scalar broadcasts to all 8 tokens; or pass 8 comma-separated floats (e.g. `1.0,0.9,1.1,...`). |
| `--summary-bias V` | `0.0` | Same format as `--summary-scale`, added in normalized space. |

### Live keyboard controls

While the stream is playing, single keypresses adjust the engine in real
time. Press `?` at any moment to print the up-to-date help in the terminal.

**Global**

| Key | Action |
| --- | --- |
| `q` | Quit. |
| `?` | Print the full keyboard-control help. |
| `r` | Reset everything to defaults (including summary & channel controls). |

**Sampler**

| Key | Action |
| --- | --- |
| `1` / `2` / `3` | Set diffusion steps to 4 / 8 / 16 NFE. |
| `e` / `h` | Switch solver to `euler` / `heun`. |
| `<` / `>` | Decrease / increase temperature by 0.1. |
| `+` / `-` | Increase / decrease context window by 4 chunks. |

**Seeds & morphing**

| Key | Action |
| --- | --- |
| `x` | Crossfade to a new random seed. |
| `X` | Hard cut to a new random seed. |
| `s` / `S` | Crossfade / hard-cut to a specific seed (type the digits, then Enter). |
| `a` | Toggle auto-cycle (periodically swap seeds). |
| `A` | Set the auto-cycle interval in chunks. |
| `[` / `]` | Decrease / increase crossfade length by 1 chunk. |

**Summary-latent control** (per-chunk, 8 tokens)

| Key | Action |
| --- | --- |
| `b` / `B` | Decrease / increase summary bias uniformly across all 8 tokens. |
| `g` / `G` | Decrease / increase summary scale uniformly. |
| `i` | Enter per-token edit mode (e.g. type `b3 0.5` to set bias of token 3). |
| `o` | Reset summary controls only (`scale=1`, `bias=0`). |
| `n` / `N` | Randomize summary bias / scale uniformly in their valid range. |
| `m` | Randomize *both* summary scale and bias. |

**Channel-latent control** (per-chunk, 64 feature dims)

| Key | Action |
| --- | --- |
| `y` / `Y` | Randomize channel bias / scale (independent per dim). |
| `u` | Randomize *both* channel scale and bias. |
| `U` | Reset channel controls only. |

### How it works (at a glance)

- The model generates one **codec chunk** per inference call. Each chunk is
  decoded to ~0.683 s of 48 kHz stereo audio.
- A **sliding window** of the last `--context-chunks` chunks is fed back as
  context, so the model has memory of recent material.
- A **prebuffer** of `--prebuffer` chunks is rendered before audio starts;
  after that, generation runs in lockstep with playback. If your machine
  generates faster than real-time, the engine throttles to avoid growing the
  buffer unbounded; if it's slower, you'll see underruns logged.
- Seed switching is **crossfaded** in latent space across `--crossfade-chunks`
  chunks so transitions don't click.

### Tips & troubleshooting

- **Audio underruns / glitches**: lower `--nfe`, switch to `--solver euler`,
  reduce `--context-chunks`, or increase `--prebuffer`.
- **Output sounds noisy or unstable**: try `--temperature 0.9` or lower, or
  use `--solver heun --nfe 8` for higher-quality steps.
- **Engine quits immediately with "checkpoint not found"**: pass an absolute
  path to `--ckpt`, or run the command from the repo root.
- **No sound**: confirm your default output device is correct
  (the engine uses `sounddevice`'s default). On macOS, you can change the
  default output in `System Settings > Sound`.
- **CoreML keeps falling back to PyTorch**: this is expected — the traced
  CoreML graph has a fixed shape, but realtime uses a variable sliding
  window. CoreML is most useful for fixed-shape batch inference.

## CoreML Support

CoDiCodec-Flow includes experimental CoreML support for inference on Apple Silicon. CoreML can leverage the Apple Neural Engine (ANE) for potentially more power-efficient inference.

### Converting a Checkpoint to CoreML

To convert a trained checkpoint to CoreML format:

```bash
# Install coremltools first
pip install coremltools

# Convert the checkpoint
python cli.py convert-coreml \
    --ckpt runs/v3_okachihuali/ema.pt \
    --out runs/v3_okachihuali/model.mlpackage \
    --use-ema \
    --context-chunks 32 \
    --min-deployment-target macos13
```

**Important Limitations:**
- CoreML models are traced with fixed input shapes. The `--context-chunks` parameter determines the sequence length used during conversion.
- Realtime audio generation uses variable sequence lengths (sliding window), which CoreML does not support natively. The CoreML backend will automatically fall back to PyTorch MPS when shapes don't match.
- For this reason, **CoreML is not recommended for realtime generation**. It may be useful for:
  - Batch inference with fixed shapes
  - iOS/macOS app deployment where battery efficiency is critical
  - Experimentation and performance comparison

### Using CoreML for Inference

To use the CoreML model for inference (with automatic fallback to PyTorch):

```bash
python -m flow.realtime \
    --ckpt runs/v3_okachihuali/ema.pt \
    --use-ema \
    --coreml-path runs/v3_okachihuali/model.mlpackage \
    --device mps \
    --solver euler \
    --nfe 4
```

If CoreML inference fails (e.g., due to shape mismatch), the system automatically falls back to PyTorch MPS backend.

### Using CoreML in the GUI App

The macOS GUI app includes a CoreML toggle in the HUD header:

1. Convert your checkpoint to CoreML format (see above)
2. Launch the GUI app
3. Click the COREML toggle in the top-right HUD
4. The app will automatically derive the CoreML path from the selected model (replaces `.pt` with `.mlpackage`)
5. Click RESTART to apply the CoreML backend

The CoreML preference is saved and persists across app restarts. If CoreML inference fails, it automatically falls back to PyTorch MPS.

### Performance Considerations

- **MPS (PyTorch)**: Recommended for most use cases, especially realtime generation. Supports dynamic shapes and is well-optimized for Apple Silicon GPUs.
- **CoreML**: May offer better power efficiency for batch inference with fixed shapes. Not suitable for the dynamic sliding-window pattern used in realtime generation.
- **Fallback**: The implementation includes automatic fallback from CoreML to PyTorch, ensuring compatibility even if CoreML fails.

## Audio Examples

Sample audio generated by codicodec-flow is available in the `examples/` directory, demonstrating the progression of the v3_okachihuali model during training:

- `okachihuali_v3_step_000000.wav` - Generated at 0 training steps (initialization)
- `okachihuali_v3_step_100000.wav` - Generated at 100,000 training steps
- `okachihuali_v3_step_200000.wav` - Generated at 200,000 training steps
- `okachihuali_v3_step_300000.wav` - Generated at 300,000 training steps
- `okachihuali_v3_step_400000.wav` - Generated at 400,000 training steps
- `okachihuali_v3_step_500000.wav` - Generated at 500,000 training steps
- `okachihuali_v3_step_600000.wav` - Generated at 600,000 training steps

The v3_okachihuali model was trained for approximately 700,000 steps on the **Okachihuali** dataset - a 60-track album by hexorcismos available at [https://hexorcismos.bandcamp.com/album/--2](https://hexorcismos.bandcamp.com/album/--2). This dataset provides a diverse collection of musical material for training the generative model.

These examples demonstrate the model's ability to generate coherent musical continuations from unconditional generation.

## Credits

**CoDiCodec-Flow Architecture**
- Moisés Horta Valenzuela, 2026

**CoDiCodec**
- The upstream CoDiCodec encoder/decoder is released by Sony CSL Paris under CC BY-NC 4.0
- Paper: Pasini et al., 2025 - [CoDiCodec: UNIFYING CONTINUOUS AND DISCRETE COMPRESSED
REPRESENTATIONS OF AUDIO](https://arxiv.org/pdf/2509.09836)
- Original repository: [https://github.com/sony/codicodec](https://github.com/sony/codicodec)

**License**
- This repository is licensed under CC BY-NC 4.0
- Code under `codicodec/` is released under CC BY-NC 4.0 by Sony CSL Paris
- The `flow/` code is under the same license unless stated otherwise
