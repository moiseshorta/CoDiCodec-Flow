# AGENT_NOTES — CoDiCodec-Flow: what changed, why, what we measured, and what the trained model taught us

This document is written for an AI agent (or a human engineer) who picks up this
repository cold and needs to (a) understand every change made relative to the
original codebase (git commit `cffa296`, "Update requirements.txt"), (b) trust or
distrust each claim based on how it was measured, (c) understand the model shipped
in `weights/`, and (d) know what to do next and which traps to avoid.

Everything below is either measured on the development machine (an NVIDIA GB10 /
DGX Spark, aarch64, CUDA 13, torch 2.14.1+cu130) or explicitly labelled as an
estimate, a design argument, or untested. Where a claim was made and later
retracted during development, the retraction is recorded too, because an agent
re-deriving results will otherwise repeat the same mistake.

---

## 0. TL;DR

* **The codebase got smaller and faster.** 21 files changed, +1067 / −2616
  lines. Three files deleted entirely (`flow/coreml_utils.py`, `tui_monitor.py`,
  `flow/data/lat_stats.py`). Same model architecture, same parameter names: old
  checkpoints load and produce identical outputs (max relative difference 3e-7).
* **Training throughput: ~1.9× faster** on the original (joint) objective at the
  default 97M model (3.43 → 6.66 steps/s at batch 8, measured with the GPU
  otherwise idle). An earlier "7.1×" figure was wrong (the old code had been
  measured while CPU-contended) and must not be quoted.
* **Correctness fixes that change what the model learns or hears:**
  1. preprocessing used to insert an ~86 ms silent dropout every `--max-seconds`
     (default 60 s) of a long file; the model learned those dropouts. Fixed.
  2. Heun / RK4 samplers evaluated the model at t = 0 in their last step, where
     the model's velocity was measured to be 87–98 % wrong. Fixed (final Euler step).
  3. Validation measured a joint-denoising loss that did not correspond to how
     audio is actually generated; it now measures *next-chunk loss* (the realtime
     condition), comparable across objectives and runs.
* **New training objective:** `--objective teacher_forcing` (block diffusion /
  "clean context → one noisy chunk" for every chunk of every crop). Learns the
  generation condition much faster; overfits small data unless the data is
  augmented (`preencode --offsets N`).
* **Inference:** K/V caching of the clean context for chunk-by-chunk generation;
  one unified solver integrator. Big win on CPU (1.9–4×), little on the GB10 GPU
  (that path is kernel-launch-bound, not FLOP-bound).
* **Shipped model (`weights/ema.pt`)**: 96.9M FlowDiT, teacher forcing, 140k steps,
  trained on FSL10K loops (~80 h incl. offset copies) + 2000 Suno full songs
  (~121 h). Final held-out next-chunk loss **1.0248**; best 1.0236 at 94k. The val
  curve has been flat since ~60–70k steps: **the model is data-limited, not
  step-limited**. More steps on this data will not help.
* **Key diagnostic insight:** train/val gap is ~1 % at low noise (timbre/texture
  generalizes) and ~31 % at high noise (the model memorizes "what comes next" in
  its training audio). Musical continuation/structure is the weak point, not
  sound quality per se.
* **Critical batch size ≈ 112 crops** (gradient noise scale at step 16.5k). On
  this GPU, which saturates at batch 16, batch 32 is the best compromise.

---

## 1. Repository map (after the changes)

```
cli.py                    thin pass-through dispatcher: preprocess | train | sample | realtime
flow/config.py            dataclasses: ModelConfig, CFMConfig, DataConfig, TrainConfig
flow/codec_wrapper.py     CoDiCodec encode/decode wrapper (+ frame_overlap property)
flow/data/preencode.py    audio -> latent shards (.pt), segmenting, offset / LR-swap augmentation
flow/data/latent_dataset.py  in-memory latent dataset, crops gathered on device, split by source
flow/model/dit.py         FlowDiT: block-causal DiT, AdaLN-Zero, RoPE, K/V cache support
flow/model/cfm.py         losses (joint, teacher_forcing), time sampling, schedules, solvers,
                          generate_chunk (cached), generate (sliding window), sample_joint
flow/model/ema.py         GPU EMA with warmup
flow/train.py             training loop, validation (next-chunk loss), previews, checkpoints
flow/sample.py            offline generation to .wav (joint or chunked mode), shared load_model()
flow/realtime.py          live streaming generation (sounddevice), keyboard controls
flow/server.py            server front-end for realtime (CoreML flag removed)
flow/smoke_test.py        equivalence tests (see §2.12)
flow/utils.py             masks (block_causal_mask, teacher_forcing_mask), logging, device helpers
codicodec/                upstream CoDiCodec package (pip install -e ./codicodec); its
                          checkpoint codicodec.pt auto-downloads from Hugging Face on first use
weights/                  the trained model + logs (this release)
```

Core shapes, for orientation:

* Codec: 48 kHz stereo; one *chunk* = 32768 samples = 0.683 s; each chunk = 8
  summary tokens × 64 dims. Latents are stored as `[T, 8, 64]` per file in
  "atanh / 0.8" space (roughly unit Gaussian).
* Model sequence: tokens flattened to `[B, L, 64]`, L = n_chunks × 8.
  Attention is bidirectional within a chunk, causal across chunks.
* Default training crop: 512 tokens = 64 chunks ≈ 43.7 s. Teacher forcing
  doubles the sequence (clean + noisy = 1024 tokens), exactly `max_seq_len`.
* Flow convention: t = 1 is pure noise, t = 0 is data,
  `x_t = (1 − t)·x + t·ε`, model predicts velocity `v = ε − x`. Sampling
  integrates from t = 1 to t = 0.

---

## 2. Every change, in detail

For each item: **Before** → **After** → **Why** → **Evidence / how verified** →
**Compatibility notes**.

### 2.1 `flow/model/dit.py` — per-chunk conditioning, positions, K/V cache

**Before.** The AdaLN conditioning vector `c = cond_mlp(timestep_embedding(t))`
was computed *per token* (`[B, L, C]`) even though every token in a chunk shares
the same noise level t. Every block then computed its AdaLN modulation (a
`Linear(C, 6·D)`) for every token. Attention had no notion of externally supplied
positions and no K/V reuse.

**After.**
* `FlowDiT.forward(x_t, t, attn_mask=None, pos=None, kv=None, return_kv=False)`.
* Conditioning is computed once per chunk: `c = cond_mlp(timestep_embedding(t[:, ::bs]))`,
  shaped `[B, n_chunks, 1, C]`; `DiTBlock.forward` reshapes `x` to
  `[B, n_chunks, bs, D]` and broadcasts the modulation across the 8 tokens of a
  chunk. Mathematically identical (t is constant within a chunk), ~30 % fewer
  FLOPs in the modulation path and far less activation memory.
* `FactorizedRoPE.forward(x, pos)` indexes `self.cos[pos]` / `self.sin[pos]`, so a
  caller can supply arbitrary token positions. Teacher forcing uses this to give
  the clean copy and the noisy copy of the crop *the same* positions
  (`pos = arange(L).repeat(2)`).
* `SelfAttention.forward(x, attn_mask, pos, kv)` returns `(out, (k, v))` and, if
  `kv` is passed, concatenates cached keys/values in front of the new ones. With
  `return_kv=True` the model returns per-layer K/V for its input, which the
  sampler caches for the clean context.

**Why.** Speed (training and inference), and enabling the teacher-forcing
objective and cached generation without architectural changes.

**Evidence.** `smoke_test.py` asserts cached-context velocity == full-window
velocity, and teacher-forced chunk i == the same chunk generated from its clean
prefix (both within 1e-4 relative). An old-code checkpoint loaded into the new
code gives the same outputs to 3e-7 relative.

**Compatibility.** Parameter names unchanged → every pre-existing checkpoint
loads with `strict=True`.

### 2.2 `flow/model/cfm.py` — losses, solvers, generation

**Before.** One loss (`cfm_loss`, "joint": a clean prefix plus a target span
denoised jointly at one shared noise level), a monolithic `sample()` with
several solver-specific helpers (`_sample_rk`, `_sample_dpmpp`,
`_sample_pingpong`, `_apply_target_update`), and chunk-by-chunk generation that
re-ran the full context window through the model at every solver step.

**After.**
* `cfm_loss(...)` — the joint objective, semantics unchanged (including the
  min-SNR renormalisation).
* `teacher_forcing_loss(model, x_data, t_chunk, attn_mask, min_snr_gamma=None, eps=None)`
  — new, see §3.1. `eps` lets validation use fixed noise.
* `min_snr_weight`, `sample_t` (uniform or logit-normal), `build_schedule`
  (linear or logSNR-shifted).
* `integrate(x, ts, velocity, solver, temperature)` — one integrator for all
  solvers: `euler`, `heun`, `midpoint`, `rk4`, `dpmpp` (DPM-Solver++ 2M for RF),
  `pingpong` (stochastic).
* **Final-step fix:** Heun and RK4 take a plain Euler step for the last interval
  (ending at t = 0) instead of evaluating the model at t = 0.
* `generate_chunk(model, context, n_steps, solver, temperature, schedule, schedule_shift, noise)`
  — runs the clean context once with `return_kv=True`, then every solver step
  only processes the 8 new tokens against the cached K/V.
* `generate(model, prefix, n_chunks, context_chunks=32, **kw)` — sliding-window
  chunk-by-chunk generation built on `generate_chunk` (what `realtime` does).
* `sample_joint(model, prefix, n_tokens, ...)` — whole-continuation joint
  denoising (what the joint objective trains).

**Why the final-step fix.** The training distribution of t never includes
exactly 0 with a meaningful target; we measured the model's velocity at t = 0 to
be 87–98 % wrong (relative error vs the ideal velocity). Heun's corrector at the
last step averaged that garbage in. The audible effect is subtle (texture, hiss
in tails); see the benchmark page section 3 in the dev repo.

**Why caching.** Chunked generation previously cost
`n_steps × (context + 8)` tokens per chunk; now it costs `context + n_steps × 8`.
Measured: 1.9–4× faster on CPU; on the GB10 GPU barely faster, because at batch
1 the GPU is bound by kernel launches, not FLOPs. (An agent optimizing GPU
realtime latency next should look at CUDA graphs / `torch.compile`, not FLOPs.)

### 2.3 `flow/model/ema.py` — EMA on the GPU with warmup

**Before.** Shadow weights lived on the CPU in fp32; every `update()` copied
every parameter GPU → CPU (a full device sync every step), and decay was a
constant 0.9999 from step 0 (so early EMA weights were ~random init for
thousands of steps).

**After.** Shadow on the model's device, updated with a single
`torch._foreach_lerp_` call; decay warmup `min(decay, (1 + n) / (10 + n))`;
the update counter is stored in the state dict as `__num_updates__`. Loading an
old EMA state (no counter) sets the counter to 10**9, i.e. "warmup done", so old
runs resume exactly as before.

**Evidence.** Part of the 1.9× training speedup; EMA previews are meaningful from
the first preview instead of after ~10k steps. A float32 vs float64 decay
comparison bug was found and fixed during testing.

### 2.4 `flow/data/latent_dataset.py` — in-memory dataset, proper crop sampling, split by source

**Before.** A torch `Dataset` + `DataLoader` with worker processes. Each
`__getitem__` picked a *file* and then a random crop inside it, so a 20-minute
file and a 45-second file were sampled equally often (long files heavily
under-represented per second of audio). Shards were scanned with mmap.
Train/val split was by shard file, so augmented variants of the same recording
could land on both sides (leakage).

**After.** `LatentDataset(pieces, crop_tokens, device, name)` holds every piece
in memory *on the training device* (1 h of audio ≈ 11 MB of latents; the full
200 h dataset fits in ~2.3 GB) and gathers crops with tensor indexing. Crop start
positions are enumerated over all pieces, so sampling is uniform **per second of
audio**. `sample(B)` returns `[B, crop, 64]` already on the device;
`fixed_crops(n)` returns evenly spaced deterministic crops (validation, previews).
`.mean` / `.std` give per-channel statistics, baked into the model as buffers.
`load_splits(root, crop, val_frac, seed, device)` splits by the shard's recorded
`source` path, so all variants (`_o1`, `_lr`, ...) of a recording stay on the
same side. Pieces shorter than a crop are dropped and the amount is logged.
`--num-workers` is gone (nothing to tune).

**Known limitation.** The split key is the *source file*. Datasets where
different files are near-duplicates (e.g. the Suno `_0` / `_1` pairs: two
generations of the same prompt) can still leak across the split. Group by a
prompt/track id if that matters.

### 2.5 `flow/data/preencode.py` — gap-free segmenting, resampling first, augmentation

**Before.** Long files were split every `--max-seconds` (README example: 60 s)
and each segment was encoded independently. The codec's STFT framing at segment
edges produced an ~86 ms near-silent dropout (measured −66 dB) at every
boundary. Resampling happened per segment.

**After.** `encode_wave(codec, wv, max_seconds)` resamples the whole file first,
then cuts segments **on the codec's chunk grid** with an overlap of
`frame_overlap = (fac − 1) · hop = 1024` samples (exposed by
`codec_wrapper.py`), and stitches the latents. Result: identical to encoding the
file in one go (verified in the smoke test, tolerance 2e-2 in tanh space;
atanh amplifies rounding near ±1). Default `--max-seconds` is now 300.

New augmentation flags (`encode_file(..., offsets, swap_lr)`, `variant_paths`):
* `--offsets N` also encodes the file shifted by 1/N, 2/N, … of a chunk. The
  codec's chunk grid then falls at different musical positions, giving different
  latents for the same music (N× data). Files are written as `<name>_o<k>.pt`.
* `--swap-lr` also encodes left/right-swapped audio (`_lr`). Useless for mono
  sources (FSL10K is effectively mono).

**Evidence.** Benchmark section 1: the old pipeline's dropout at 8.1 s is clearly
audible in an encode→decode round trip; the new one has none (quietest point
−8 dB). Any latents encoded with the old code have these gaps: re-encode with
`--overwrite`.

### 2.6 `flow/train.py` — the training loop

**Before.** DataLoader workers; `.item()` calls every step (GPU→CPU syncs); CPU
EMA; validation = joint loss on a prefix grid; `torch.load(weights_only=False)`;
non-atomic checkpoint writes; audio previews always via the joint sampler.

**After.**
* Data from `load_splits` (on device), no workers.
* `build_model(cfg, lat_mean, lat_std)` bakes per-channel latent normalization
  into the checkpoint (SD3 / Stable Audio style) so data matches N(0, I) noise.
* `--objective {joint, teacher_forcing}`; `--t-sample-mode {uniform, logit_normal}`;
  `--min-snr-gamma`; `--dropout`; architecture flags.
* Fused AdamW on CUDA; optional `--compile` (untested here, see §6).
* No per-step GPU syncs: the running loss is accumulated on device and only
  read every `--log-every` steps.
* tqdm progress bar with `logging_redirect_tqdm`, log lines with loss, lr, steps/s.
* **Validation = next-chunk loss** (`evaluate`): 16 fixed val crops, every chunk
  denoised from its clean preceding chunks (teacher-forcing layout regardless of
  the training objective), a fixed grid of noise levels t ∈ {0.1, 0.3, 0.5, 0.7,
  0.9} rotated so each chunk sees every level, seeded noise. Reproducible across
  calls, runs and objectives, and it measures exactly the realtime condition.
* Previews: joint objective → `sample_joint`; teacher forcing → `generate`
  (chunk by chunk, like realtime). `--audio-continuation` continues fixed val
  prompts instead of unconditional generation.
* Checkpoints: `_atomic_save` (write `.tmp`, then `os.replace`), `last.pt`
  (model + optimizer + EMA + step + config) and `ema.pt` (EMA + config + step);
  loaded with `weights_only=True`.
* Auto-resume from `out_dir/last.pt`; `--init-from` for fine-tuning (weights
  only, fresh optimizer, fresh dataset statistics, EMA seeded from source).

**Evidence.** 3.43 → 6.66 steps/s on the joint objective (1.9×), batch 8, 97M
model, GPU idle otherwise. Teacher forcing at batch 8: 2.84 steps/s (it processes
2× the tokens).

### 2.7 `flow/sample.py`

Shared `load_model(ckpt, device, use_ema=True)` used by sample/realtime/server
(rebuilds the architecture from `config.model` in the checkpoint, dropout forced
to 0). New flags: `--mode joint|chunked` (joint: whole continuation at once, must
fit the ~87 s window with the prompt; chunked: like realtime, any length),
`--context-chunks`, `--temperature`, `--seed`, `--schedule`, `--schedule-shift`.
The output wav contains prompt + continuation so the seam is audible.

**Agent note:** for a teacher-forcing model (like the shipped one) use
`--mode chunked`; that is the condition it was trained for. `joint` mode still
works but is out of distribution for this model.

### 2.8 `flow/realtime.py` and `flow/server.py`

* Uses `generate_chunk` (cached context).
* CoreML path removed (`flow/coreml_utils.py` deleted; it was Mac-only and
  unmaintained relative to the model code).
* PortAudio import failure (`OSError` from `sounddevice` when `libportaudio2` is
  missing) is caught with a clear message instead of a traceback.
* Adaptive output queue: `throttle_to = max(2.0, (crossfade_chunks + 1) · chunk_t + 0.5)`
  seconds, so the generator never runs further ahead than needed for the
  crossfade, keeping keyboard controls responsive.
* `server.py`: CoreML flag removed, otherwise unchanged.

### 2.9 `flow/utils.py`

* `teacher_forcing_mask(L, bs, device)`: `[2L, 2L]` additive float mask (0 = allowed, −inf = blocked). Clean half is
  block-causal over itself; noisy chunk i attends to clean chunks < i and to
  itself (its own 8 tokens), nothing else.
* `get_logger` attaches a single handler to the parent `flow` logger (no more
  duplicated log lines when modules are imported in different orders).

### 2.10 `flow/config.py`

Defaults now: `objective="joint"` (safe default for small data),
`t_sample_mode="uniform"`, `crop_tokens=512`, `ckpt_every=1000`,
`audio_sample_every=0` (previews off unless asked), `audio_nfe=16`,
`compile=False`; `num_workers` removed.

### 2.11 `cli.py`

Was a ~300-line file duplicating every subcommand's argparse. Now a
pass-through dispatcher: `python cli.py <subcommand> [args]` runs the module's
own parser (`flow.data.preencode`, `flow.train`, `flow.sample`, `flow.realtime`).
One source of truth for flags.

### 2.12 `flow/smoke_test.py`

Runs in seconds on a tiny randomized model (weights re-initialised with
std 0.05 because AdaLN-Zero starts as identity, which would make equivalence
checks trivially pass). Checks:
1. forward + backward for both objectives;
2. cached-context velocity == full-window velocity;
3. teacher-forced chunk == generation condition;
4. every solver produces the right shape and finite values, shifted schedule,
   unconditional generation, joint sampling;
5. `--with-codec`: segmented encoding == whole-file encoding, decode round trip.

Run `python -m flow.smoke_test` after any change to model/cfm/utils.

### 2.13 Deleted / dependencies / docs

* Deleted: `flow/coreml_utils.py` (326 lines), `tui_monitor.py` (327 lines,
  superseded by the tqdm bar + log lines), `flow/data/lat_stats.py` (131 lines,
  statistics are now computed in `LatentDataset` and stored in the checkpoint).
* `requirements.txt`: + `sounddevice`, − `pypickle`.
* `README.md`, `flow/README.md`, Colab notebook updated with the new flags,
  measured results and the teacher-forcing + offsets recommendation.

---

## 3. Concepts an agent must understand

### 3.1 Teacher forcing (block diffusion) vs joint

* **Joint** (original): pick a prefix length; prefix is clean; the rest of the
  crop is noised with **one** t and denoised jointly. Only the first target
  chunk sees the exact realtime condition (clean context → one noisy chunk);
  later target chunks see *noisy* preceding chunks, which never happens in
  realtime generation.
* **Teacher forcing**: input `[clean crop ; noisy crop]` (2L tokens), positions
  repeated, `teacher_forcing_mask`. Each noisy chunk i has its **own** t and
  attends to clean chunks < i plus itself. Every chunk of every crop trains the
  exact realtime condition. Cost: 2× tokens per crop.
* Small-data test (1.6 h FSL10K, 20M model, 3k steps, next-chunk loss on an
  unseen recording): joint 1.434, joint + offsets 1.422, TF 1.469 (overfits),
  **TF + offsets 1.385 (best)**, and TF + offsets reached joint's final loss within
  1k steps. → Use TF with augmentation; keep joint as the default for unaugmented
  tiny datasets.

### 3.2 Next-chunk loss is the number to watch

It is the loss of the exact operation realtime performs. It is comparable
between objectives, but **not between different validation splits**: adding data
re-draws the split (see §4.3), so a new series starts.

### 3.3 Reading per-noise-level losses

* Low t (0.1): the chunk is nearly clean; the model only fixes fine detail →
  measures timbre/texture modelling.
* High t (0.9): the chunk is nearly pure noise; the model must infer the chunk
  from context → measures musical prediction ("what comes next").

---

## 4. The trained model in `weights/`

### 4.1 Data

* **FSL10K** (Freesound Loop Dataset): 9486 loops, 44.1 kHz, effectively mono.
  Stitched into 127 files of ~20 min each (to give the model long, crop-able
  sequences), resampled to 48 kHz by preencode, encoded with `--offsets 2`
  (2 variants per file). ≈ 80.7 h train / 4.0 h val in phase 1.
* **cn_part18_of_25**: 2000 Suno-generated full songs (`suno_cn_0170xx_{0,1}.mp3`,
  48 kHz stereo, avg ~218 s, ≈ 121 h). Encoded **without** offsets (only the first
  ~30 files got an offset copy before the switch; harmless). Pairs `_0`/`_1` are
  two generations of the same prompt.
* Phase-2 dataset as logged: **train 2196 pieces, 198.85 h; val 112 pieces,
  9.51 h**; 5 + 1 pieces shorter than a crop dropped.
* Unique content is much less than the hours suggest: offset copies are the same
  music; loops repeat every bar. Rough unique content: FSL10K ≈ 40 h minus loop
  repetition, Suno ≈ 121 h (and the `_0`/`_1` pairs share style/prompt).

### 4.2 Recipe and timeline

Common: FlowDiT 96.9M (dim 768, 10 layers, 12 heads × 64, mlp ×4, cond 768,
block 8, max_seq_len 1024), teacher forcing, uniform t, AdamW β = (0.9, 0.95),
wd 0.01, grad clip 1.0, EMA 0.9999 with warmup, bf16 autocast, crop 512 tokens,
dropout 0.1, cosine LR with 2000-step warmup, previews every 2000 steps (3 val
prompts, 4 s real + 16 s generated, Heun 16 steps, EMA weights).

| Phase | Steps | Data | Batch (micro × accum) | Peak LR | Throughput |
|---|---|---|---|---|---|
| 1a | 0 → 14 000 | FSL10K + offsets | 16 (8 × 2) | 3e-4 | 2.85 steps/s ≈ 46 crops/s |
| 1b | 14 000 → 17 000 | same | 128 (64 × 2) | 5e-4 (cosine over 40k) | 0.36 steps/s ≈ 46 crops/s |
| 2 | 17 000 → 140 000 | FSL10K + Suno | 32 (32 × 1) | 2.5e-4 (cosine over 140k) | **0.71 steps/s ≈ 23 crops/s** |

Phase 2 took ~48 h wall-clock (expected ~24 h). **Phase-2 throughput was half
that of phase 1 per crop and stayed constant from start to end; the cause was
not investigated.** Candidates: something else on the GPU/unified memory, a
batch-32-specific kernel choice, or memory pressure from the larger in-GPU
dataset. An agent continuing on this box should benchmark batch 16/32/64 for 2
minutes each before a long run.

Phase boundaries were resumes from `last.pt` (optimizer state carried over; the
cosine schedule is recomputed from the new `--max-steps`, so LR jumps at each
boundary). Normalization buffers (`lat_mean`/`lat_std`) are the **FSL10K
statistics from phase 1**; they were not recomputed when Suno songs were added.

### 4.3 Validation results

Phase 1 (FSL10K split, 16 val crops):

| step | next-chunk loss |
|---|---|
| 2 000 | 1.2867 |
| 6 000 | 1.1737 |
| 10 000 | 1.1436 |
| 14 000 | 1.1268 (end of bs 16) |
| 14 500 | 1.1154 (bs 128) |
| 16 000 | 1.1062 |
| 16 500 | 1.1064 (plateau) |

Phase 2 (new split incl. Suno; **not comparable to phase 1**):

| step | next-chunk loss |
|---|---|
| 18 000 | 1.0877 |
| 20 000 | 1.0696 |
| 30 000 | 1.0411 |
| 40 000 | 1.0330 |
| 50 000 | 1.0275 |
| 60 000 | 1.0248 |
| 66 000 | 1.0237 |
| 70 000 | 1.0240 |
| 94 000 | **1.0236 (best)** |
| 100 000 | 1.0248 |
| 120 000 | 1.0248 |
| 140 000 | 1.0248 (final) |

Interpretation: ~95 % of phase-2 gains happened in the first ~40k steps; after
~60k the curve is flat within the measurement noise of a 16-crop validation
(differences of ±0.001 are noise). The LR annealing to 0 at the end did not
produce the usual end-of-cosine improvement, which is further evidence the model
is data/capacity-limited on this mixture, not optimization-limited. The
`ema.pt` shipped is the final (step 140k) EMA; the best-val step (94k) is not
materially better.

### 4.4 Diagnostic at step 16 500 (phase 1, `weights/logs/diag.json`)

Same evaluation for train and val: 64 fixed crops each, per noise level, seeded
noise.

| t | train (raw) | val (raw) | gap |
|---|---|---|---|
| 0.1 | 1.145 | 1.157 | +1 % |
| 0.3 | 1.209 | 1.277 | +6 % |
| 0.5 | 1.093 | 1.208 | +10 % |
| 0.7 | 0.864 | 1.028 | +19 % |
| 0.9 | 0.621 | 0.813 | +31 % |
| mean | 0.986 | 1.097 | +11 % |

EMA vs raw: val 1.0947 (EMA) vs 1.0966 (raw); train 1.003 (EMA) vs 0.986 (raw).
EMA slightly improves generalization → always generate with EMA weights.

Gradient noise scale (McCandlish et al. 2018, "An Empirical Model of
Large-Batch Training", appendix A estimator with batches 16 and 128, 30 reps,
dropout on): |G|² = 0.001265, tr(Σ) = 0.1414, **B_crit ≈ 112 crops**.

Implications: steps-to-target ∝ 1 + B_crit/B, crops-to-target ∝ 1 + B/B_crit.
At B = 16: 8.0× min steps, 1.14× min crops. At B = 32: 4.5×, 1.29×. At B = 128:
1.9×, 2.14×. **On hardware that is saturated at small batch (this GB10 at
batch 16), wall-clock ∝ crops, so small-to-moderate batches win**; batch 128
costs ~1.9× more GPU time per unit of progress and does more passes over the
data per unit of progress (worse for overfitting). B_crit grows as loss falls,
so this is a snapshot; treat it as "50–200".

### 4.5 Insights about the model (what to expect when you listen)

1. **Timbre and texture are its strength.** Low-noise loss generalizes almost
   perfectly; generated audio sounds codec-clean, with correct instrument colour.
2. **Long-range musical structure is its weakness.** The high-noise gap means it
   learned the specific continuations of its training audio rather than general
   rules; on new prompts continuations are plausible in timbre but vaguer in
   rhythm/harmony development, and can drift.
3. **Style prior is skewed toward the Suno material** (~60 % of phase-2 hours,
   and full songs give far more non-repetitive tokens than loops). Expect
   song-like (vocals, Mandarin-pop-style production) output more often than
   loop-like output when unconditioned.
4. **Loop prompts** (FSL10K-like): the model tends to continue the loop
   convincingly for a few bars because it saw ~90 passes over loop data in
   phase 1; novelty over time is limited.
5. **Normalization mismatch:** buffers come from FSL10K only. If outputs from
   song prompts sound level-shifted or noisier, re-compute stats on the full set
   and fine-tune briefly (`--init-from` does recompute them).
6. **Use `--mode chunked` / `realtime`**, Heun, 8–16 steps. The model was trained
   only on the chunked condition.
7. **Data, not steps, is the bottleneck.** ~160 h of mostly-unique audio for 97M
   parameters is small by audio-diffusion standards (Stable Audio Open used
   ~7300 h for 1.1B). Rough estimate: 500–2000 h unique audio before a 97M model
   stops plateauing early. This is an estimate, not a measurement.

---

## 5. Lessons learned during development (read before making claims)

* **Measure throughput with the machine idle.** The first speedup measurement
  (7.1×) was inflated because the old code ran while the CPU was contended.
  The corrected figure is 1.9×. Re-run both sides back to back on an idle box.
* **GPU realtime ≠ FLOPs.** Batch-1 generation on the GB10 is launch-bound; the
  K/V cache's 3–8× FLOP reduction gave little wall-clock gain there.
* **Teacher forcing overfits faster** because every chunk is a training target.
  Always pair with augmentation or more data, and watch next-chunk loss.
* **Bigger batch did not train faster** on this GPU: crops/s was identical at
  16 and 128. Only use big batches if the hardware has spare parallelism.
* **Val split changes when data is added** (split by shuffled source list). Keep
  the old series separate (this run kept `samples_phase1/`, `reference_phase1/`,
  `train_phase1b_bs128.log` on the dev machine) or pin the val set.
* **Augmentation economics:** `--offsets 2` doubles encode time. For full songs
  (non-repetitive) it is less valuable than for loops; for phase 2 we skipped it
  to halve encoding (5 h → 2.5 h at ~17 s per 3.6-min song per process, 4
  processes).
* **Concurrent jobs on unified memory:** rendering audio while training while a
  large LLM server held ~98 GB caused OOM. Run one GPU job at a time, or check
  `free -g` first.

---

## 6. Environment gotchas (dev box: NVIDIA GB10 / DGX Spark)

* Use a venv with torch 2.14.1+cu130 (sm_121 support); `pip install -e ./codicodec`.
* `python3.12-dev` is missing → Triton/`torch.compile` fails → `train --compile`
  is untested. Install it (sudo) before trying.
* `libportaudio2` missing → `flow.realtime` / `flow.server` cannot play audio.
  Install it (sudo apt install libportaudio2) for live playback.
* No ffmpeg/ffprobe; `soundfile` reads mp3 directly (libsndfile ≥ 1.1).
* Shell scripting traps hit during this work:
  - `pgrep -f pattern` / `pkill -f pattern` match the invoking shell's own
    command line if it contains the pattern → loops that never end or a shell
    that kills itself. Use `ps aux | grep "[p]attern"` or anchor the pattern
    (`pgrep -f "^/full/path/python -m flow.train"`).
  - A launcher that does `wait` on background encoders proceeds as soon as the
    encoders exit **for any reason** — kill the launcher before killing its
    children, or it will start training on partial data.

**Other platforms (checked 2026-11-20 on an Apple M4 Max, macOS 15.6, torch 2.11):**

* MPS: `smoke_test --with-codec` passes; `train` runs both objectives with bf16
  autocast, validation, checkpointing and resume (synthetic latent shards). Default
  97M model, teacher forcing, batch 8: ≈ 0.4 steps/s (GB10: 2.84). Fused AdamW and
  GradScaler are enabled on CUDA only (`train.py`).
* Windows + CUDA: not run. The preprocess/train path has no POSIX-only code and never
  calls the codec's `torch.compile`'d encoder (`fix_batch_size=False`).
  `flow/realtime.py` imports `termios`/`tty`, so `realtime` does not start on
  Windows (already true before these changes).

---

## 7. How to use the shipped weights

```bash
pip install -r requirements.txt && pip install -e ./codicodec
python -m flow.smoke_test                         # wiring check, no codec needed

# continuation of a prompt, chunk by chunk (the trained condition)
python cli.py sample --ckpt weights/ema.pt --prompt-wav prompt.wav \
    --duration-s 30 --mode chunked --nfe 8 --solver heun --out out.wav

# unconditional
python cli.py sample --ckpt weights/ema.pt --duration-s 30 --mode chunked --out uncond.wav

# live
python cli.py realtime --ckpt weights/ema.pt
```

Fine-tune on new audio (recomputes normalization stats, fresh optimizer):

```bash
python cli.py preprocess --in-dir my_audio --out-dir data/my_latents --offsets 2
python cli.py train --data-dir data/my_latents --out-dir runs/ft \
    --init-from weights/last.pt --objective teacher_forcing \
    --batch-size 32 --grad-accum 1 --lr 1e-4 --warmup-steps 1000 --max-steps 30000 \
    --val-every 2000 --ckpt-every 2000 --audio-sample-every 2000 --audio-continuation
```

Resume this exact run with more steps: put `weights/last.pt` into an out-dir as
`last.pt`, pass the same data and a larger `--max-steps` (note: LR schedule is
recomputed; at step 140k of a longer schedule LR jumps back up).

---

## 8. Recommended next steps (ranked by expected impact)

1. **More genuinely new data.** Pitch-shift (±1, ±2 semitones) and time-stretch
   (±8 %) variants *before* encoding; other datasets; real (non-generated) full
   tracks. Target ≥ 500 h unique.
2. **Right-size the model to the data.** For ~160 h, compare from scratch:
   97M vs ~30M (`--dim 512 --n-layers 8 --n-heads 8 --cond-dim 512`). Expect the
   smaller one to match val loss and overfit later. Untested.
3. **Fix the split leakage** for paired data: group sources by track/prompt id in
   `load_splits` (e.g. strip the `_0`/`_1` suffix) and pin the val set across
   data additions so loss series stay comparable.
4. **Recompute normalization stats** on the full mixture (fine-tune with
   `--init-from` does this).
5. **Regularization targeted at high-t memorization:** dropout 0.2; consider
   context dropout (randomly truncate the clean context) so the model relies
   less on exact continuations. Untested.
6. **Investigate the phase-2 throughput halving** (§4.2) before the next long run.
7. **Classifier-free guidance** is not wired, though chunk 0 of every crop is
   trained without context, so an unconditional branch exists; with the K/V cache
   it costs only 8 extra tokens per solver step.
8. **GPU realtime latency:** CUDA graphs / `torch.compile` for `generate_chunk`
   (requires python3.12-dev on this box).
9. **LR sweep** with the branch tool in `weights/logs/study.py`
   (`study.py branch --lr ... --batch ...`): branches from a checkpoint with
   optimizer state, constant LR then linear decay over the last 25 % (so arms are
   compared *annealed*, avoiding the "lower LR always looks better short-term"
   trap), and evaluates on fixed val crops. It expects `base.pt` next to it and
   the dataset at `data/fsl10k_latents`; edit paths as needed. Not run.

---

## 9. What was *not* verified

* `--compile` (no python3.12-dev on the dev box).
* Anything on Windows (see §6, "Other platforms").
* Live audio playback via `realtime` on the dev box (no PortAudio); the engine
  was timed with `sounddevice` stubbed.
* The model-size and dropout recommendations (no from-scratch comparisons were
  run on the full dataset).
* Any learning-rate optimality: 3e-4 / 5e-4 / 2.5e-4 were reasoned choices, not
  swept.
* The cause of the phase-2 throughput drop.
* Listening-quality claims beyond the per-noise-level loss analysis: §4.5 points
  1–4 are inferences from the losses and the data mix; the training previews
  in the dev repo (`runs/fsl10k_tf/samples/`, 189 clips) were not formally
  evaluated.

## 10. File provenance in this zip

* Code: the working tree of the dev repo (tracked files with all modifications,
  plus untracked non-ignored files such as `synth/`), excluding `data/`, `runs/`,
  `venv/` and the codec checkpoint `codicodec/codicodec/models/codicodec.pt`
  (auto-downloaded from Hugging Face on first use; needs internet once).
* `examples/*.wav`: pre-existing demo clips from the original repository (the
  older v3_okachihuali model), referenced by the README. Not dataset audio.
* `weights/ema.pt`, `weights/last.pt`: step 140 000 of this run.
* `weights/logs/train_bs16.log` (phase 1a), `train_phase1b_bs128.log` (phase 1b),
  `train.log` (phase 2), `diag.json` (§4.4), `study.py` (diagnostic + branch tool).
* `weights/README.md`: short model card.
* Git repository (unlike this zip): `weights/*.pt` and `weights/logs/*.log` are not
  committed (GitHub's 100 MB file limit, `*.log` ignore rule); the checkpoints are
  kept private.
