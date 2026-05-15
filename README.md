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

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./codicodec    # makes the upstream package importable

# 2. Verify codec works on your machine (downloads checkpoint on first run)
python -m flow.smoke_test

# 3. Pre-encode an audio folder to latent shards
python -m flow.data.preencode \
    --in-dir   ~/music/training \
    --out-dir  ./data/latents   \
    --device   mps

# 4. Train a tiny continuation model
python -m flow.train \
    --data-dir ./data/latents   \
    --out-dir  ./runs/v0        \
    --device   mps

# 5. Generate a continuation of a prompt audio file
python -m flow.sample \
    --ckpt        ./runs/v0/ema.pt \
    --prompt-wav  ./prompt.wav     \
    --duration-s  20               \
    --out         ./out.wav        \
    --device      mps
```

## Roadmap

- [x] v0: block-causal CFM continuation model + offline sampling.
- [ ] v1: streaming inference with KV-cache + `sounddevice` realtime demo.
- [ ] v2: classifier-free guidance via unconditional prompt dropout.
- [ ] v3: continuous control knobs (density / brightness) via AdaLN.
- [ ] v4: consistency / MeanFlow distillation for 1-2 NFE inference.

## License

Code under `codicodec/` is released under CC BY-NC 4.0 by Sony CSL Paris.
The `flow/` code is under the same license unless stated otherwise.
