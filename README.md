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
```

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
