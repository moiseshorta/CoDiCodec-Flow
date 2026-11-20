"""Hyperparameter study branched from a training checkpoint.

diag:   train vs val loss (raw + EMA, per noise level) and gradient noise scale (critical batch).
branch: continue from the checkpoint with one setting changed for a fixed budget of crops,
        constant LR then linear decay to 0 over the last 25% (so every arm is compared annealed),
        and report val loss on the same fixed crops/noise for all arms.
"""
import argparse, json, math, sys, time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from flow.config import ModelConfig
from flow.data.latent_dataset import load_splits
from flow.model.cfm import sample_t, teacher_forcing_loss
from flow.model.dit import FlowDiT
from flow.model.ema import EMA
from flow.utils import autocast_ctx, teacher_forcing_mask

OUT = Path(__file__).resolve().parent
dev = torch.device("cuda")
T_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


def load(ckpt, dropout=None):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    mc = ModelConfig(**sd["config"]["model"])
    if dropout is not None:
        mc.dropout = dropout
    model = FlowDiT(**asdict(mc))
    model.load_state_dict(sd["model"])
    return model.to(dev), sd


@torch.no_grad()
def evaluate(model, crops, mask, mb=16):
    """Per-noise-level teacher-forced loss on fixed crops with seeded noise."""
    model.eval()
    gen = torch.Generator().manual_seed(0)
    nc = crops.shape[1] // model.block_size
    out = {}
    for t in T_GRID:
        eps = torch.randn(crops.shape, generator=gen).to(dev)
        tot = 0.0
        for i in range(0, len(crops), mb):
            tc = torch.full((len(crops[i:i + mb]), nc), t, device=dev)
            with autocast_ctx(dev):
                tot += float(teacher_forcing_loss(model, crops[i:i + mb], tc, mask, eps=eps[i:i + mb])) * len(crops[i:i + mb])
        out[t] = tot / len(crops)
    out["mean"] = sum(out[t] for t in T_GRID) / len(T_GRID)
    model.train()
    return out


def grad_sq(model, ds, mask, n, mb=16):
    """Squared norm of the mean gradient over n random crops."""
    model.zero_grad(set_to_none=True)
    nc = ds.crop_tokens // model.block_size if hasattr(ds, "crop_tokens") else None
    for _ in range(n // mb):
        x = ds.sample(mb)
        tc = sample_t(mb * (x.shape[1] // model.block_size), "uniform", dev).view(mb, -1)
        with autocast_ctx(dev):
            loss = teacher_forcing_loss(model, x, tc, mask)
        (loss * mb / n).backward()
    return sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None)


def diag(a, train_ds, val_ds, mask):
    model, sd = load(a.base)
    res = {"step": sd["step"]}
    tr_c, va_c = train_ds.fixed_crops(64), val_ds.fixed_crops(64)
    res["raw_train"], res["raw_val"] = evaluate(model, tr_c, mask), evaluate(model, va_c, mask)
    ema = EMA(model, decay=0.0); ema.load_state_dict(sd["ema"])
    em, _ = load(a.base); ema.copy_to(em)
    res["ema_train"], res["ema_val"] = evaluate(em, tr_c, mask), evaluate(em, va_c, mask)
    del em
    # Gradient noise scale (McCandlish et al. 2018, App. A), dropout on as in training.
    model.train()
    bs, bb, gs, gb = 16, 128, [], []
    t0 = time.time()
    for r in range(a.reps):
        gs.append(grad_sq(model, train_ds, mask, bs)); gb.append(grad_sq(model, train_ds, mask, bb))
    gs_m, gb_m = sum(gs) / len(gs), sum(gb) / len(gb)
    G2 = (bb * gb_m - bs * gs_m) / (bb - bs)
    S = (gs_m - gb_m) / (1 / bs - 1 / bb)
    res.update(G2=G2, S=S, B_crit=S / G2, grad_reps=a.reps, grad_secs=time.time() - t0)
    print(json.dumps(res, indent=1))
    (OUT / "diag.json").write_text(json.dumps(res, indent=1))


def branch(a, train_ds, val_ds, mask):
    model, sd = load(a.base, a.dropout)
    opt = AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd, eps=1e-8, fused=True)
    opt.load_state_dict(sd["optim"])
    for g in opt.param_groups:
        g["weight_decay"] = a.wd
    va_c = val_ds.fixed_crops(64)
    steps = a.crops // (a.batch * a.accum)
    curve, t0 = [], time.time()
    torch.manual_seed(1)
    model.train()
    for s in range(steps):
        frac = s / steps
        lr = a.lr * (1.0 if frac < 0.75 else (1 - frac) / 0.25)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(a.accum):
            x = train_ds.sample(a.batch)
            tc = sample_t(a.batch * (x.shape[1] // model.block_size), "uniform", dev).view(a.batch, -1)
            with autocast_ctx(dev):
                loss = teacher_forcing_loss(model, x, tc, mask)
            (loss / a.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (s + 1) % max(1, steps // 4) == 0 or s + 1 == steps:
            v = evaluate(model, va_c, mask)
            curve.append({"step": s + 1, "crops": (s + 1) * a.batch * a.accum, "val": v["mean"],
                          "train_loss": float(loss), "secs": round(time.time() - t0)})
            print(a.name, curve[-1], flush=True)
    res = {"name": a.name, "lr": a.lr, "batch": a.batch * a.accum, "dropout": a.dropout, "wd": a.wd,
           "crops": a.crops, "final_val": curve[-1]["val"], "final_val_per_t": v, "curve": curve}
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["diag", "branch"])
    p.add_argument("--base", default=str(OUT / "base.pt"))
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--name", default="arm")
    p.add_argument("--lr", type=float, default=3.3e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--crops", type=int, default=64000)
    a = p.parse_args()
    train_ds, val_ds = load_splits(str(REPO / "data/fsl10k_latents"), 512, val_frac=0.05, seed=42, device=dev)
    mask = teacher_forcing_mask(512, 8, dev)
    (diag if a.phase == "diag" else branch)(a, train_ds, val_ds, mask)
