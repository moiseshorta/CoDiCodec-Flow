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

  # Indefinite real-time streaming generation (press 'q' to quit)
  python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema --device mps
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
    train_parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps")
    train_parser.add_argument("--dtype", default="bf16", help="Data type (bf16, fp32)")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers")
    train_parser.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    train_parser.add_argument("--val-every", type=int, default=1000, help="Validate every N steps")
    train_parser.add_argument("--ckpt-every", type=int, default=1000, help="Checkpoint every N steps")
    train_parser.add_argument("--audio-sample-every", type=int, default=0, help="Generate audio samples every N steps (0 to disable)")
    train_parser.add_argument("--audio-n-samples", type=int, default=2, help="Number of audio samples to generate")
    train_parser.add_argument("--audio-prompt-seconds", type=float, default=4, help="Audio prompt duration in seconds")
    train_parser.add_argument("--audio-continuation-seconds", type=float, default=8, help="Audio continuation duration in seconds")
    train_parser.add_argument("--audio-nfe", type=int, default=16, help="Audio sampling NFE")
    train_parser.add_argument("--audio-solver", default="heun",
                              choices=["euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong"],
                              help="Audio sampling solver during training")
    train_parser.add_argument("--audio-unconditional", action="store_true", help="Generate unconditional audio samples")
    train_parser.add_argument("--t-sample-mode", default="uniform", help="Time sampling mode (uniform, importance)")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    train_parser.add_argument("--dim", type=int, default=None, help="Model dimension")
    train_parser.add_argument("--n-layers", type=int, default=None, help="Number of layers")
    train_parser.add_argument("--n-heads", type=int, default=None, help="Number of heads")
    train_parser.add_argument("--cond-dim", type=int, default=None, help="Conditioning dimension")
    train_parser.add_argument("--init-from", default=None, help="Path to checkpoint to initialize from (for fine-tuning)")
    train_parser.add_argument("--no-tui", action="store_true", help="Disable TUI for monitoring training")
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Generate audio from a checkpoint")
    sample_parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    sample_parser.add_argument("--prompt-wav", help="Prompt audio file")
    sample_parser.add_argument("--duration-s", type=float, default=20, help="Duration in seconds")
    sample_parser.add_argument("--nfe", type=int, default=8, help="Number of function evaluations")
    sample_parser.add_argument("--solver", default="heun",
                               choices=["euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong"],
                               help="Solver: euler/heun/midpoint/rk4 (ODE), dpmpp (DPM-Solver++ 2M for RF), pingpong (SDE)")
    sample_parser.add_argument("--schedule", default="linear", choices=["linear", "shifted"],
                               help="Time grid: linear or logSNR-shifted")
    sample_parser.add_argument("--schedule-shift", type=float, default=0.0,
                               help="LogSNR shift exponent for --schedule shifted")
    sample_parser.add_argument("--out", required=True, help="Output audio file")
    sample_parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    sample_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    
    # Realtime command — indefinite streaming generation
    realtime_parser = subparsers.add_parser(
        "realtime",
        help="Real-time streaming generation (plays audio indefinitely)",
        description=(
            "Stream generated audio in real time. Runs indefinitely until the user "
            "quits with 'q' or Ctrl+C, or until --max-chunks is reached. Wraps "
            "`python -m flow.realtime`."
        ),
    )
    realtime_parser.add_argument("--ckpt", required=True, help="Path to last.pt or ema.pt")
    realtime_parser.add_argument("--use-ema", action="store_true",
                                  help="Load EMA weights (recommended for inference)")
    realtime_parser.add_argument("--device", default=None,
                                  help="cuda | mps | cpu (default: auto-detect)")
    realtime_parser.add_argument("--coreml-path", default=None,
                                  help="Optional path to CoreML .mlpackage (falls back to PyTorch on shape mismatch)")
    # Sampler
    realtime_parser.add_argument("--nfe", type=int, default=4,
                                  help="ODE steps per chunk (default: 4)")
    realtime_parser.add_argument("--solver", default="euler",
                                  choices=["euler", "heun", "midpoint", "rk4", "dpmpp", "pingpong"],
                                  help="Sampler (default: euler)")
    realtime_parser.add_argument("--temperature", type=float, default=1.0,
                                  help="Velocity scaling (<1 sharpens, >1 diffuses)")
    realtime_parser.add_argument("--seed-scale", type=float, default=0.0,
                                  help="Shrink initial noise toward 0 (default: 0)")
    # Streaming / context
    realtime_parser.add_argument("--context-chunks", type=int, default=32,
                                  help="Sliding-window context length in codec chunks (default: 32, ~21.8s @ 48kHz)")
    realtime_parser.add_argument("--prebuffer", type=int, default=2,
                                  help="Chunks to render before starting playback (default: 2)")
    realtime_parser.add_argument("--crossfade-chunks", type=int, default=4,
                                  help="Crossfade length when switching seeds (default: 4)")
    # Run control
    realtime_parser.add_argument("--max-chunks", type=int, default=None,
                                  help="Stop after N chunks (default: run until 'q')")
    realtime_parser.add_argument("--save", default=None,
                                  help="Optional path to save the full session as .wav")
    realtime_parser.add_argument("--seed", type=int, default=None,
                                  help="Initial RNG seed (default: time-based)")
    # Summary-latent control
    realtime_parser.add_argument("--summary-scale", default="1.0",
                                  help="Initial summary-latent scale: scalar or 8 comma-separated floats (default: 1.0)")
    realtime_parser.add_argument("--summary-bias", default="0.0",
                                  help="Initial summary-latent bias: scalar or 8 comma-separated floats (default: 0.0)")

    # Convert to CoreML command
    coreml_parser = subparsers.add_parser("convert-coreml", help="Convert a checkpoint to CoreML format")
    coreml_parser.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    coreml_parser.add_argument("--out", required=True, help="Output CoreML model path (.mlpackage)")
    coreml_parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA weights (default: True)")
    coreml_parser.add_argument("--context-chunks", type=int, default=32, help="Number of context chunks for tracing (default: 32)")
    coreml_parser.add_argument("--min-deployment-target", default="macos13", 
                               choices=["macos13", "ios16", "ios17", "macos14"],
                               help="Minimum deployment target (default: macos13)")
    
    args = parser.parse_args()
    
    if args.command == "preprocess":
        import flow.data.preencode
        sys.argv = ["preencode", "--in-dir", args.in_dir, "--out-dir", args.out_dir, "--device", args.device, "--max-seconds", str(args.max_seconds)]
        flow.data.preencode.main()
    elif args.command == "train":
        if not args.no_tui:
            import tui_monitor
            tui_monitor.launch_tui_train(
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                device=args.device,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                crop_tokens=args.crop_tokens,
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
                dtype=args.dtype,
                lr=args.lr,
                num_workers=args.num_workers,
                log_every=args.log_every,
                val_every=args.val_every,
                ckpt_every=args.ckpt_every,
                audio_sample_every=args.audio_sample_every,
                audio_n_samples=args.audio_n_samples,
                audio_prompt_seconds=args.audio_prompt_seconds,
                audio_continuation_seconds=args.audio_continuation_seconds,
                audio_nfe=args.audio_nfe,
                audio_solver=args.audio_solver,
                audio_unconditional=args.audio_unconditional,
                t_sample_mode=args.t_sample_mode,
                dropout=args.dropout,
                dim=args.dim,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                cond_dim=args.cond_dim,
                init_from=args.init_from,
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
                "--warmup-steps", str(args.warmup_steps),
                "--dtype", args.dtype,
                "--lr", str(args.lr),
                "--num-workers", str(args.num_workers),
                "--log-every", str(args.log_every),
                "--val-every", str(args.val_every),
                "--ckpt-every", str(args.ckpt_every),
                "--audio-sample-every", str(args.audio_sample_every),
                "--audio-n-samples", str(args.audio_n_samples),
                "--audio-prompt-seconds", str(args.audio_prompt_seconds),
                "--audio-continuation-seconds", str(args.audio_continuation_seconds),
                "--audio-nfe", str(args.audio_nfe),
                "--audio-solver", args.audio_solver,
                "--t-sample-mode", args.t_sample_mode,
                "--dropout", str(args.dropout),
            ]
            if args.init_from:
                train_args.extend(["--init-from", args.init_from])
            if args.audio_unconditional:
                train_args.append("--audio-unconditional")
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
            "--schedule", args.schedule,
            "--schedule-shift", str(args.schedule_shift),
            "--duration-s", str(args.duration_s),
        ]
        if args.prompt_wav:
            sample_args.extend(["--prompt-wav", args.prompt_wav])
        sys.argv = sample_args
        flow.sample.main()
    elif args.command == "realtime":
        import flow.realtime
        rt_args = [
            "realtime",
            "--ckpt", args.ckpt,
            "--nfe", str(args.nfe),
            "--solver", args.solver,
            "--temperature", str(args.temperature),
            "--seed-scale", str(args.seed_scale),
            "--context-chunks", str(args.context_chunks),
            "--prebuffer", str(args.prebuffer),
            "--crossfade-chunks", str(args.crossfade_chunks),
            "--summary-scale", args.summary_scale,
            "--summary-bias", args.summary_bias,
        ]
        if args.use_ema:
            rt_args.append("--use-ema")
        if args.device:
            rt_args.extend(["--device", args.device])
        if args.coreml_path:
            rt_args.extend(["--coreml-path", args.coreml_path])
        if args.max_chunks is not None:
            rt_args.extend(["--max-chunks", str(args.max_chunks)])
        if args.save:
            rt_args.extend(["--save", args.save])
        if args.seed is not None:
            rt_args.extend(["--seed", str(args.seed)])
        sys.argv = rt_args
        flow.realtime.main()
    elif args.command == "convert-coreml":
        import flow.coreml_utils
        success = flow.coreml_utils.convert_checkpoint_to_coreml(
            ckpt_path=args.ckpt,
            output_path=args.out,
            use_ema=args.use_ema,
            context_chunks=args.context_chunks,
            min_deployment_target=args.min_deployment_target,
        )
        if success:
            print(f"Successfully converted {args.ckpt} to CoreML format: {args.out}")
            sys.exit(0)
        else:
            print(f"Failed to convert {args.ckpt} to CoreML format")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
