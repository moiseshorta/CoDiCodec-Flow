#!/usr/bin/env python3
"""
CLI wrapper for codicodec-flow training, preprocessing, and generation.

This provides a user-friendly command-line interface to the flow package
with TUI monitoring for training metrics.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="codicodec-flow: Block-causal Flow Matching DiT for CoDiCodec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess audio data
  python cli.py preprocess --in-dir ~/music/training --out-dir ./data/latents --device mps

  # Train a model
  python cli.py train --data-dir ./data/latents --out-dir ./runs/v0 --device mps

  # Generate audio
  python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --out ./out.wav --device mps
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess audio data to latent shards")
    preprocess_parser.add_argument("--in-dir", required=True, help="Input audio directory")
    preprocess_parser.add_argument("--out-dir", required=True, help="Output latent directory")
    preprocess_parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    preprocess_parser.add_argument("--max-seconds", type=int, default=300, help="Max seconds per file")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", required=True, help="Data directory with latent shards")
    train_parser.add_argument("--out-dir", required=True, help="Output directory for checkpoints")
    train_parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    train_parser.add_argument("--crop-tokens", type=int, default=512, help="Crop tokens")
    train_parser.add_argument("--max-steps", type=int, default=200000, help="Maximum training steps")
    train_parser.add_argument("--dtype", default="bf16", help="Data type (bf16, fp32)")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--dim", type=int, default=None, help="Model dimension")
    train_parser.add_argument("--n-layers", type=int, default=None, help="Number of layers")
    train_parser.add_argument("--n-heads", type=int, default=None, help="Number of heads")
    train_parser.add_argument("--cond-dim", type=int, default=None, help="Conditioning dimension")
    train_parser.add_argument("--tui", action="store_true", help="Enable TUI for monitoring training")
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate audio from a checkpoint")
    sample_parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    sample_parser.add_argument("--prompt-wav", help="Prompt audio file")
    sample_parser.add_argument("--duration-s", type=float, default=20, help="Duration in seconds")
    sample_parser.add_argument("--nfe", type=int, default=8, help="Number of function evaluations")
    sample_parser.add_argument("--solver", default="heun", help="Solver (euler, heun)")
    sample_parser.add_argument("--out", required=True, help="Output audio file")
    sample_parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    sample_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        import flow.data.preencode
        sys.argv = ["preencode", "--in-dir", args.in_dir, "--out-dir", args.out_dir, "--device", args.device, "--max-seconds", str(args.max_seconds)]
        flow.data.preencode.main()
    elif args.command == "train":
        if args.tui:
            import tui_monitor
            tui_monitor.launch_tui_train(
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                device=args.device,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                crop_tokens=args.crop_tokens,
                max_steps=args.max_steps,
                dtype=args.dtype,
                lr=args.lr,
                dim=args.dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                cond_dim=args.cond_dim,
            )
        else:
            import flow.train
            train_args = [
                "train",
                "--data-dir", args.data_dir,
                "--out-dir", args.out_dir,
                "--device", args.device,
                "--batch-size", str(args.batch_size),
                "--grad-accum", str(args.grad_accum),
                "--crop-tokens", str(args.crop_tokens),
                "--max-steps", str(args.max_steps),
                "--dtype", args.dtype,
            ]
            if args.dim:
                train_args.extend(["--dim", str(args.dim)])
            if args.n_layers:
                train_args.extend(["--n-layers", str(args.n_layers)])
            if args.n_heads:
                train_args.extend(["--n-heads", str(args.n_heads)])
            if args.cond_dim:
                train_args.extend(["--cond-dim", str(args.cond_dim)])
            sys.argv = train_args
            flow.train.main()
    elif args.command == "sample":
        import flow.sample
        sample_args = [
            "sample",
            "--ckpt", args.ckpt,
            "--out", args.out,
            "--device", args.device,
            "--nfe", str(args.nfe),
            "--solver", args.solver,
            "--temperature", str(args.temperature),
            "--duration-s", str(args.duration_s),
        ]
        if args.prompt_wav:
            sample_args.extend(["--prompt-wav", args.prompt_wav])
        sys.argv = sample_args
        flow.sample.main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
