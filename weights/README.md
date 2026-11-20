# FlowDiT weights: FSL10K + Suno songs (teacher forcing)

> Not distributed with this repository: the checkpoint files below and the raw
> `logs/*.log`. This card, `logs/diag.json` and `logs/study.py` are included.

| File | What | Use |
|---|---|---|
| `ema.pt` (388 MB) | EMA weights, step 140000 | inference (`sample`, `realtime`, `server`) |
| `last.pt` (1.5 GB) | raw weights + optimizer + EMA, step 140000 | resume / fine-tune (`--init-from weights/last.pt`) |

## Model
FlowDiT, 96.9M params: dim 768, 10 layers, 12 heads × 64, block size 8, max 1024 tokens
(~87 s window). Latent normalization stats are stored in the checkpoint.
The CoDiCodec checkpoint is not included: it downloads automatically on first use.

## Training recipe
Objective: teacher forcing (every chunk denoised from its clean preceding chunks, the
exact realtime generation condition). Uniform t, AdamW (0.9, 0.95), wd 0.01, grad clip 1.0,
EMA 0.9999, bf16, crop 512 tokens (~44 s), dropout 0.1, cosine LR.

| Phase | Steps | Data | Batch | Peak LR |
|---|---|---|---|---|
| 1a | 0 – 14k | FSL10K, 80.7 h (offset augmented ×2) | 16 | 3e-4 |
| 1b | 14k – 17k | same | 128 | 5e-4 |
| 2 | 17k – 140k | FSL10K + 2000 Suno songs, 198.9 h train / 9.5 h val | 32 | 2.5e-4 |

~48 h of phase 2 on an NVIDIA GB10. Logs in `logs/`.

## Results
Next-chunk validation loss (held-out sources, phase-2 split):
1.049 (26k) → 1.029 (46k) → 1.024 (66k) → **1.025 at 140k**: flat since ~70k steps,
so more steps on this data will not help; more data or a smaller model would.

`logs/diag.json` (phase 1, step 16.5k): train/val gap concentrated at high noise levels
(memorized continuations, timbre generalizes); gradient-noise critical batch ≈ 112.

## Quick start
```bash
pip install -r requirements.txt && pip install -e ./codicodec
python cli.py sample --ckpt weights/ema.pt --prompt-wav prompt.wav --duration-s 20 --out out.wav
python cli.py realtime --ckpt weights/ema.pt
```
