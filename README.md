# codicodec-flow

A generative model that synthesises audio in CoDiCodec's continuous latent space
using Conditional Flow Matching (CFM) on a block-causal DiT architecture.

The model targets **musical continuation / improvising accompaniment**: given a
short audio prompt, it generates an arbitrarily long continuation in a
chunk-causal, streaming fashion on the codec's ~11.7 Hz, 64-channel latent
sequence.

## Why this design

- **CoDiCodec** ([Pasini et al., 2025](https://arxiv.org/pdf/2509.09836))
  encodes 48 kHz stereo audio to summary embeddings at ~11.7 Hz with 64
  channels (128x compression) and exposes a streaming `decode_next()` API.
  The continuous latents, after the codec's `atanh / sigma_rescale=0.8`
  transform, are approximately unit-Gaussian — a direct fit for flow
  matching.
- **Block-causal Flow Matching DiT** is the simplest architecture that:
  1. respects the codec's chunk structure (8 latent tokens per ~0.683 s
     chunk, permutation-invariant within a chunk),
  2. supports KV-caching for efficient streaming inference,
  3. has unconditional dropout-based classifier-free guidance for free.
- The whole pipeline is **MPS-friendly** so it can be trained and run in
  real-time on a 36 GB Apple Silicon laptop.

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

## Preprocessing

Before training, you need to convert your audio files into latent shards using the CoDiCodec encoder.

```bash
python -m flow.data.preencode \
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
python -m flow.train \
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
python -m flow.train --data-dir ./data/latents --out-dir ./runs/v0 \
    --device mps --batch-size 4 --grad-accum 2 --crop-tokens 512 \
    --dtype bf16 --max-steps 200000
```

Smaller (~20M params, faster iteration):
```bash
python -m flow.train --data-dir ./data/latents --out-dir ./runs/v0 \
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

**Monitoring:**
```bash
# View training logs
tensorboard --logdir ./runs/v0
```

## Generation

Generate audio continuations using a trained checkpoint.

### Continuation from a prompt

```bash
python -m flow.sample \
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
- `--solver`: ODE solver: `euler` (faster) or `heun` (better quality)
- `--out`: Output audio file path
- `--device`: `mps`, `cuda`, or `cpu`
- `--temperature`: Sampling temperature (default: 1.0, higher = more diverse)
- `--n-steps`: Number of diffusion steps (default: 32)

### Unconditional generation

```bash
python -m flow.sample \
    --ckpt        ./runs/v0/ema.pt \
    --duration-s  20               \
    --nfe         8                \
    --solver      heun             \
    --out         ./out_uncond.wav \
    --device      mps
```

Omit `--prompt-wav` for unconditional generation (no prompt context).

### Advanced options

```bash
# Higher quality with more sampling steps
python -m flow.sample \
    --ckpt ./runs/v0/ema.pt \
    --prompt-wav ./prompt.wav \
    --duration-s 30 \
    --nfe 16 \
    --solver heun \
    --out ./out_high_quality.wav \
    --device mps

# Faster generation with fewer steps
python -m flow.sample \
    --ckpt ./runs/v0/ema.pt \
    --prompt-wav ./prompt.wav \
    --duration-s 20 \
    --nfe 4 \
    --solver euler \
    --out ./out_fast.wav \
    --device mps

# Adjust temperature for diversity
python -m flow.sample \
    --ckpt ./runs/v0/ema.pt \
    --prompt-wav ./prompt.wav \
    --duration-s 20 \
    --nfe 8 \
    --solver heun \
    --temperature 1.5 \
    --out ./out_diverse.wav \
    --device mps
```

**Sampling Trade-offs:**
- **NFE (steps)**: More steps = better quality but slower. 4-8 is real-time, 16+ is high quality.
- **Solver**: Heun is more accurate than Euler but ~2x slower.
- **Temperature**: Higher values increase diversity but may reduce coherence.

## Audio Examples

Sample audio generated by codicodec-flow is available in the `examples/` directory, demonstrating the progression of the v3_okachihuali model during training:

- `okachihuali_v3_step_1000000.wav` - Generated at 1,000,000 training steps
- `okachihuali_v3_step_2000000.wav` - Generated at 2,000,000 training steps
- `okachihuali_v3_step_3400000.wav` - Generated at 3,400,000 training steps
- `okachihuali_v3_step_5000000.wav` - Generated at 5,000,000 training steps
- `okachihuali_v3_example.wav` - Generated at 6,860,000 training steps (final model)

The v3_okachihuali model was trained on the **Akachihuali** dataset - a 60-track album by hexorcismos available at [https://hexorcismos.bandcamp.com/album/--2](https://hexorcismos.bandcamp.com/album/--2). This dataset provides a diverse collection of musical material for training the generative model.

These examples demonstrate how the model's output quality improves over training steps, with noticeable improvements in coherence, musicality, and fidelity as training progresses.

## Credits

**codicodec-flow Architecture**
- Moisés Horta Valenzuela, 2026

**CoDiCodec**
- The upstream CoDiCodec encoder/decoder is released by Sony CSL Paris under CC BY-NC 4.0
- Paper: Pasini et al., 2025 - [CoDiCodec: Codec-based Audio Generation with Discrete Latents](https://arxiv.org/pdf/2509.09836)
- Original repository: [https://github.com/sony/codicodec](https://github.com/sony/codicodec)

**License**
- This repository is licensed under CC BY-NC 4.0
- Code under `codicodec/` is released under CC BY-NC 4.0 by Sony CSL Paris
- The `flow/` code is under the same license unless stated otherwise
