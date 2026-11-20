# `flow/` — Block-causal Flow-Matching DiT for CoDiCodec

This is the core package of `codicodec-flow`. It contains the model, training
loop, and inference code; the upstream codec lives in `../codicodec/`.

## High-level design

```
audio (~48kHz stereo)
   │
   │   CoDiCodec encoder
   ▼
continuous latents [T_chunks * 8, 64]   (~ 11.7 Hz × 64 channels, ~unit Gaussian)
   │
   │   FlowDiT (block-causal, AdaLN-Zero, CFM v-prediction)
   ▼
denoised continuation latents
   │
   │   CoDiCodec decoder (parallel or autoregressive)
   ▼
audio (~48kHz stereo)
```

### Why block-causal?

Each ~0.683 s audio chunk is encoded into 8 **summary** tokens. Summary
tokens are permutation-invariant within a chunk (CoDiCodec paper, §4.1) but
chunks are temporally ordered. The most natural attention pattern is therefore:

- **Bidirectional** within a chunk.
- **Causal** across chunks.

This matches the codec's own decoder mask, enables chunk-wise KV caching at
inference (the clean context runs once per generated chunk; the solver steps
only run the new chunk), and gives the model a direct one-chunk-ahead
inductive bias for continuation.

### Training objectives

`joint` (default): a clean prefix plus a target span denoised jointly at one
noise level. Only the first target chunk of each crop sees the realtime
condition (clean context -> one noisy chunk).

`teacher_forcing` (`--objective teacher_forcing`): the model sees
`[clean crop ; noisy crop]` and noisy chunk i attends to clean chunks < i and
to itself (`utils.teacher_forcing_mask`), each chunk with its own noise level,
so every chunk trains the realtime condition. In a small test (1.6 h of audio,
20M model, 3k steps, next-chunk loss on an unseen recording) it was best at
every checkpoint on `preprocess --offsets 2` data (1.385 vs 1.422 for `joint`),
but overfit without augmentation (1.469 vs 1.434): pair it with `--offsets`.

### Why Conditional Flow Matching?

- The codec's continuous latents (after `atanh / sigma_rescale`) are roughly
  unit-Gaussian, which is exactly the assumption CFM with the rectified-flow
  path makes.
- CFM training is stable, single-loss, and free of adversarial dynamics.
- Few-step Heun/Euler sampling (4–8 NFE) gets us real-time inference on
  modest hardware.
- Easy to distill later (consistency / MeanFlow) for 1–2 NFE if needed.

## Files

| File | Role |
| --- | --- |
| `codec_wrapper.py` | MPS-safe wrapper around `codicodec.EncoderDecoder` (patches the upstream `mixed_precision` flag for non-CUDA devices, exposes encode/decode helpers). |
| `config.py` | Dataclass-based config for model / CFM / data / training. |
| `utils.py` | Device picker, attention masks (block-causal, teacher forcing), autocast context, logging. |
| `data/preencode.py` | CLI: scans an audio directory, encodes with CoDiCodec, writes per-file `.pt` shards (optional offset / L-R augmentation). |
| `data/latent_dataset.py` | `load_splits`: all shards in memory, random crops gathered on device, train/val split by source recording. |
| `model/dit.py` | `FlowDiT` block-causal transformer with per-chunk AdaLN-Zero conditioning and K/V caching. |
| `model/cfm.py` | Training losses (joint, teacher forcing), time samplers, solvers, joint and chunk-by-chunk sampling. |
| `model/ema.py` | EMA of parameters on the model's device, with warmup. |
| `train.py` | Training loop entry point. |
| `sample.py` | Offline sampling entry point (+ shared `load_model`). |
| `realtime.py` | Live streaming generation with keyboard controls. |
| `server.py` | stdio JSON backend wrapping `realtime` for an external GUI. |
| `smoke_test.py` | Tiny end-to-end sanity check (incl. cache / teacher-forcing / encoding equivalence). |

## End-to-end usage

```bash
# 0. Verify the wiring on your hardware (no codec download needed)
python -m flow.smoke_test

# 1. Pre-encode a directory of audio
python -m flow.data.preencode \
    --in-dir   ~/datasets/music \
    --out-dir  ./data/latents

# 2. Train (a few hundred K steps; logs every 50)
python -m flow.train \
    --data-dir ./data/latents \
    --out-dir  ./runs/v0      \
    --batch-size 8 --max-steps 200000

# 3a. Unconditional generation (no prompt)
python -m flow.sample \
    --ckpt        ./runs/v0/ema.pt \
    --duration-s  20               \
    --nfe         8 --solver heun  \
    --out         ./out_uncond.wav

# 3b. Continuation of a 4s prompt (decoder same as 3a, just adds --prompt-wav)
python -m flow.sample \
    --ckpt        ./runs/v0/ema.pt \
    --prompt-wav  ./prompt.wav     \
    --duration-s  20               \
    --nfe         8 --solver heun  \
    --out         ./out_cont.wav
```

Periodic audio samples during training are off by default; enable them with
`--audio-sample-every N`. They are unconditional unless you pass
`--audio-continuation` (continue fixed validation clips).

## Tensor shape conventions

- `wv` waveform: `[2, N]` or `[B, 2, N]`, float32, range `[-1, 1]`.
- `latent` (codec output, with `desired_channels=64`): `[T, 8, 64]` per file
  (or `[T*8, 64]` flattened for the model). T is the number of chunks
  (0.683 s each at 48 kHz).
- `x_data` for the model: `[B, L, 64]` with `L = crop_tokens` (must be a
  multiple of 8 = `block_size`).
- `t` for the model: `[B, L]` float in `[0, 1]`, constant within each chunk.
  Clean context chunks get `t=0`; with teacher forcing each noisy chunk has
  its own `t`.

## Defaults (and how to change them)

The shipped config (`config.py`) targets ~100M params (~97M precisely) and is
designed to fit on a 36 GB Apple Silicon machine with `batch_size=4`,
`crop_tokens=512`, bf16 autocast:

| Knob | Default | Notes |
| --- | --- | --- |
| `dim` | 768 | transformer hidden dim |
| `n_layers` | 10 | transformer depth |
| `n_heads` × `head_dim` | 12 × 64 | must equal `dim` |
| `mlp_mult` | 4 | SwiGLU expansion |
| `cond_dim` | 768 | per-chunk AdaLN-Zero modulation (≈39% of params) |
| `max_seq_len` | 1024 tokens | ≈ 87 s of audio |
| `crop_tokens` | 512 | 64 chunks, ≈ 44 s of training context |
| `objective` | joint | or `teacher_forcing` |
| `batch_size` × `grad_accum` | 8 × 2 | effective batch 16 (lower to 4×2 on MPS) |
| `dtype` | bf16 | falls back to fp32 if MPS bf16 op unsupported |
| `lr` | 1e-4 with 2k warmup | cosine to 0 over `max_steps` |
| `ema_decay` | 0.9999 | applied after every optim step |
| `p_unconditional` | 0.1 | `joint` only: random "no prefix" rate |
| `prefix_min_frac` / `prefix_max_frac` | 0.05 / 0.75 | `joint` only: random prefix span |

Param breakdown (≈97M total): 10 × (attn 2.36M + SwiGLU 3.54M + AdaLN-mod 3.54M)
= 94.4M in the transformer stack, plus 2.6M in input/output projections,
positional embeddings, and conditioning MLP.

A smaller ~20M variant (good for fast iteration) can be obtained by overriding
in code or by adding CLI flags:

```python
cfg.model.dim = 384
cfg.model.n_layers = 8
cfg.model.n_heads = 6
cfg.model.cond_dim = 384
```

To train at the default ~100M scale on a 36 GB Mac:

```bash
python -m flow.train --data-dir ./data/latents --out-dir ./runs/v0 \
    --batch-size 4 --grad-accum 2 --crop-tokens 512 \
    --dtype bf16 --max-steps 200000
```

## Known caveats & TODOs

- **Classifier-free guidance** is not wired. Chunk 0 of every crop is
  trained without context, so a context-free branch exists; with the K/V
  cache it costs only 8 extra tokens per solver step.
- **Continuous control knobs** (density, brightness, etc.) require an extra
  conditioning vector in the AdaLN modulation. Out of scope for v0.
- **MPS bf16 coverage** is improving but uneven. If a particular op blows
  up, fall back to `--dtype fp32`.
- **`encoder_forward_fast`** is `torch.compile`'d in the upstream package
  and won't run on MPS; the wrapper avoids it.
