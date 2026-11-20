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
- **Block-causal Flow Matching DiT**: bidirectional attention within a chunk,
  causal across chunks, matching the codec's chunk structure. Generation runs
  chunk by chunk: the clean context window goes through the model once per
  chunk (its keys/values are cached) and the solver steps only run the 8 new
  tokens.
- **Two training objectives**: `joint` (default) denoises a continuation span
  jointly after a clean prefix; `teacher_forcing` trains every chunk exactly as
  realtime generates it (clean context → one noisy chunk, each chunk with its
  own noise level). Teacher forcing learns that condition much faster; on
  small datasets pair it with augmentation (`preprocess --offsets`).
- Runs on CUDA and Apple Silicon (MPS); trains and runs in real time on a
  36 GB Apple Silicon laptop.

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
  cli.py               `python cli.py <command> ...` entry point
  codicodec/           upstream codec package (do not modify)
  flow/                this project
    __init__.py
    codec_wrapper.py   MPS-safe wrapper around codicodec.EncoderDecoder
    config.py          dataclass-based config
    utils.py           device, masks, logging
    data/
      preencode.py     audio dir -> per-file .pt latent shards
      latent_dataset.py  in-memory crops, train/val split by recording
    model/
      dit.py           block-causal flow-matching DiT (with K/V caching)
      cfm.py           training losses + samplers (joint, chunk by chunk)
      ema.py
    train.py           training loop
    sample.py          offline sampling
    realtime.py        live streaming generation
    server.py          stdio JSON backend for an external GUI
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

# Linux only, for realtime playback: the PortAudio library used by sounddevice
sudo apt install libportaudio2

# Verify the model and codec work on your machine (downloads the codec checkpoint on first run)
python -m flow.smoke_test --with-codec
```

## CLI Interface

`cli.py` forwards each command's options unchanged to the matching module
(`flow.data.preencode`, `flow.train`, `flow.sample`, `flow.realtime`), so
`python cli.py <command> -h` lists everything a command accepts. The device is
auto-detected (CUDA, then MPS, then CPU); pass `--device` to override.

```bash
# Preprocess audio data
python cli.py preprocess --in-dir ~/music/training --out-dir ./data/latents

# Train a model (progress bar + log lines)
python cli.py train --data-dir ./data/latents --out-dir ./runs/v0

# Generate audio
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --out ./out.wav

# Stream audio indefinitely with live keyboard controls (press 'q' to quit)
python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema
```

See [Real-time Streaming Generation](#real-time-streaming-generation) for the full list of options and live keyboard controls.

## Preprocessing

Before training, you need to convert your audio files into latent shards using the CoDiCodec encoder.

```bash
python cli.py preprocess \
    --in-dir   /path/to/your/audio \
    --out-dir  ./data/latents
```

**Arguments:**
- `--in-dir`: Directory containing audio files (WAV, MP3, FLAC, etc.)
- `--out-dir`: Output directory for latent shards (.pt files)
- `--device`: `mps`, `cuda` or `cpu` (default: auto)
- `--max-seconds`: Files longer than this (default: 300 s) are encoded in segments to bound memory. Segments are cut on the codec's chunk grid, so the result is identical to encoding the whole file.
- `--offsets N`: augmentation for small datasets. Also encodes each file shifted by 1/N, 2/N, ... of a chunk, which gives different latents for the same music (N× data).
- `--swap-lr`: augmentation. Also encodes left/right-swapped copies (2× data).
- `--overwrite`: re-encode files that already have a `.pt`.

**Output:**
- Each audio file produces a `.pt` file containing the encoded latent representation (plus `<name>_o<k>[_lr].pt` variants when augmenting)
- Latents are stored as `[T, 8, 64]` tensors (T = number of 0.683s chunks)
- Each shard records its source file; the train/val split keeps all variants of a recording on the same side

**Tips:**
- Use diverse audio for better generalization (different styles, instruments, tempos)
- 48 kHz stereo audio is recommended (CoDiCodec's native rate); other rates are resampled
- Aim for several hours of audio for reasonable training; on small datasets add `--offsets 4`
- Files shorter than a training crop (default ~44 s) are skipped at training time; the training log reports how much was dropped
- Latents encoded before this version have a short zero-padded gap every `--max-seconds` for files longer than that; re-encode with `--overwrite`
- Train for at least 100K steps for meaningful results; the v3_okachihuali model was trained for ~700,000 steps

## Training

Train a block-causal Flow Matching DiT model on the preprocessed latents.

```bash
python cli.py train \
    --data-dir   ./data/latents \
    --out-dir    ./runs/v0      \
    --batch-size 4              \
    --grad-accum 2              \
    --crop-tokens 512          \
    --max-steps 200000
```

**Key Arguments:**
- `--data-dir`: Directory containing preprocessed latent shards
- `--out-dir`: Output directory for checkpoints and logs
- `--device`: `mps`, `cuda`, or `cpu` (default: auto)
- `--batch-size`: Batch size per GPU (default: 8, use 4 on MPS)
- `--grad-accum`: Gradient accumulation steps (effective batch = batch_size × grad_accum)
- `--crop-tokens`: Random crop length in tokens (default: 512 ≈ 44 s, must be multiple of 8)
- `--max-steps`: Total training steps (default: 200000)
- `--dtype`: `bf16` for bfloat16 (faster, less memory) or `fp32` for float32
- `--lr`: Learning rate (default: 1e-4 with cosine decay)
- `--objective`: `joint` (default) is the original objective: a clean prefix plus a target span denoised jointly at one noise level. `teacher_forcing` trains every chunk the way realtime generates it (clean context → one noisy chunk, its own noise level per chunk). In a small test (1.6 h of audio, 20M model, 3k steps, `next_chunk_loss` on an unseen recording), teacher forcing on `preprocess --offsets 2` data was best at every checkpoint (1.385 vs 1.422 for `joint` on the same data) and reached `joint`'s final loss within 1k steps. Without augmentation it overfit after ~1.5k steps and ended worse (1.469 vs 1.434). So: use it together with `--offsets`, and watch `next_chunk_loss`.
- `--compile`: `torch.compile` the training forward (CUDA; about a minute of startup)
- `--t-sample-mode`: `uniform` (default) or `logit_normal`
- `--init-from`: initialize weights from another run's `last.pt` (fine-tuning; fresh optimizer and dataset stats)

All the data is loaded into memory once (1 h of audio ≈ 11 MB of latents), so
there are no data-loader workers to tune.

**Model Size Configuration:**

Default (~97M params, recommended for 36GB+ RAM):
```bash
python cli.py train --data-dir ./data/latents --out-dir ./runs/v0 \
    --batch-size 4 --grad-accum 2 --crop-tokens 512 \
    --dtype bf16 --max-steps 200000
```

Smaller (~20M params, faster iteration):
```bash
python cli.py train --data-dir ./data/latents --out-dir ./runs/v0 \
    --batch-size 8 --grad-accum 2 --crop-tokens 512 \
    --dtype bf16 --max-steps 200000 \
    --dim 384 --n-layers 8 --n-heads 6 --cond-dim 384
```

Try the teacher-forcing objective on an existing model (no architecture
change, so any earlier checkpoint loads; watch `next_chunk_loss`):
```bash
python cli.py train --data-dir ./data/latents --out-dir ./runs/v3_tf \
    --init-from ./runs/v3_okachihuali/last.pt --objective teacher_forcing
```

**Training Details:**
- Checkpoints are saved every 1000 steps (`--ckpt-every`): `last.pt` (latest, for resuming) and `ema.pt` (EMA copy, for inference). Re-running the same command resumes from `last.pt`.
- Validation reports `next_chunk_loss`: the loss of denoising each held-out chunk from its clean preceding chunks, i.e. the realtime condition. It's comparable across steps, runs and objectives.
- Audio previews are off by default; `--audio-sample-every 1000` writes clips to `out_dir/samples/` (sampled the way the objective trains: whole continuation for `joint`, chunk by chunk for `teacher_forcing`). Unconditional by default; `--audio-continuation` continues fixed validation clips instead.
- Logs include loss, learning rate, and steps/s

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
    --out         ./out.wav
```

**Arguments:**
- `--ckpt`: Path to checkpoint (use `ema.pt` for best quality, `last.pt` for latest)
- `--prompt-wav`: Audio prompt file (WAV, 48 kHz stereo recommended)
- `--duration-s`: Duration of continuation in seconds (default: 20)
- `--mode`: `joint` (default) denoises the whole continuation at once; prompt + continuation must fit the model window (~87 s). `chunked` generates chunk by chunk with a sliding context window, exactly like realtime, so any length works.
- `--nfe`: Solver steps per chunk (default: 8). Model calls per step depend on the solver, see below.
- `--solver`: Sampler (default `heun`). Supported:
  - `euler`    : 1st-order ODE, 1 model call/step (fastest baseline)
  - `heun`     : 2nd-order ODE, 2 calls/step (default; the last step is Euler)
  - `midpoint` : 2nd-order RK2, 2 calls/step
  - `rk4`      : 4th-order Runge-Kutta, 4 calls/step (the last step is Euler)
  - `dpmpp`    : DPM-Solver++ 2M for rectified flow, 1 call/step (strong at 4-8 steps)
  - `pingpong` : stochastic SDE sampler, 1 call/step (recommended for distilled / few-step models)
- `--schedule`: Time grid: `linear` (default) or `shifted` (logSNR shift)
- `--schedule-shift`: LogSNR shift exponent for `--schedule shifted` (e.g. `1.0`-`3.0`)
- `--context-chunks`: `--mode chunked` only: sliding context window in chunks (default: 32 ≈ 22 s, as in realtime)
- `--temperature`: Velocity scaling (default: 1.0; <1 sharpens, >1 leaves more noise)
- `--seed`: Fix the random seed
- `--out`: Output audio file path
- `--device`: `mps`, `cuda`, or `cpu` (default: auto)

### Unconditional generation

```bash
python cli.py sample \
    --ckpt        ./runs/v0/ema.pt \
    --duration-s  20               \
    --nfe         8                \
    --solver      heun             \
    --out         ./out_uncond.wav
```

Omit `--prompt-wav` for unconditional generation (no prompt context).

### Advanced options

**Higher quality with more sampling steps:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 30 --nfe 16 --solver heun --out ./out_high_quality.wav
```

**Faster generation with fewer steps:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 4 --solver euler --out ./out_fast.wav
```

**Adjust temperature for diversity:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 8 --solver heun --temperature 1.5 --out ./out_diverse.wav
```

**DPM-Solver++ for very low NFE:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 6 --solver dpmpp --out ./out_dpmpp.wav
```

**RK4 for highest quality at low NFE:**
```bash
python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --duration-s 20 --nfe 4 --solver rk4 --out ./out_rk4.wav
```

**Sampling Trade-offs:**
- **Steps (`--nfe`)**: More steps = better quality but slower. In realtime and `--mode chunked` each step only runs the 8 new tokens (the context is computed once per chunk), so extra steps are cheap.
- **Solver**: `euler`/`dpmpp`/`pingpong` use 1 model call per step; `heun`/`midpoint` use 2; `rk4` uses 4. Compare solvers at equal model calls, e.g. `euler --nfe 8` vs `heun --nfe 4`. `pingpong` adds stochasticity (only recommended for distilled models).
- **Schedule**: `--schedule shifted --schedule-shift 1.0` warps the time grid in log-SNR space (SD3-style); helpful at low step counts.
- **Temperature**: Higher values increase diversity but may reduce coherence.

## Real-time Streaming Generation

The `realtime` subcommand spins up a streaming generator that **plays audio
indefinitely** out of the system's default sound device while the FlowDiT
model continually fills a sliding-window buffer in the background. Use it for
headless live-coding sessions, long-form ambient playback, or stress-testing a
checkpoint.

### Quick start

```bash
# Minimal: stream from a checkpoint until you press 'q'
python cli.py realtime --ckpt ./runs/v3_okachihuali/ema.pt --use-ema

# Same thing, but called directly as a module
python -m flow.realtime --ckpt ./runs/v3_okachihuali/ema.pt --use-ema
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

```

### CLI options

| Flag | Default | Description |
| --- | --- | --- |
| `--ckpt PATH` | *required* | Checkpoint to load (`last.pt` or `ema.pt`). |
| `--use-ema` | off | Load EMA weights — strongly recommended for inference. |
| `--device DEV` | auto | `mps`, `cuda`, or `cpu`. Auto-detected if omitted. |
| `--nfe N` | `4` | Solver steps per chunk. Higher = better quality, lower = lower latency. |
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
  context, so the model has memory of recent material. The window runs through
  the model once per chunk (its keys/values are cached); the solver steps then
  only run the new chunk's 8 tokens.
- A **prebuffer** of `--prebuffer` chunks is rendered before audio starts;
  after that, generation runs in lockstep with playback. If your machine
  generates faster than real-time, the engine keeps ~3 s queued, so key
  presses are heard within ~3 s (a little more on machines that need the
  margin to render a crossfade without gaps); if it's slower, you'll see
  underruns logged.
- Each chunk prints an **RT %** metric: `audio_seconds / compute_seconds × 100`.
  Values **> 100%** mean the model is generating faster than real-time (good),
  values **< 100%** mean it can't keep up and you'll eventually hit underruns.
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
- **"PortAudio library not found" (Linux)**: install it with
  `sudo apt install libportaudio2`.

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
