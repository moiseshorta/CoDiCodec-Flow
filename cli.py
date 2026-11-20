#!/usr/bin/env python3
"""codicodec-flow CLI: `python cli.py <command> [options]`.

Each command forwards its options unchanged to the matching module, so
`python cli.py <command> -h` lists everything it accepts.

Examples:
  python cli.py preprocess --in-dir ~/music/training --out-dir ./data/latents
  python cli.py train --data-dir ./data/latents --out-dir ./runs/v0
  python cli.py sample --ckpt ./runs/v0/ema.pt --prompt-wav ./prompt.wav --out ./out.wav
  python cli.py realtime --ckpt ./runs/v0/ema.pt --use-ema
"""
import importlib
import sys

COMMANDS = {
    "preprocess": ("flow.data.preencode", "encode an audio directory to latent shards"),
    "train": ("flow.train", "train a model"),
    "sample": ("flow.sample", "generate audio from a checkpoint"),
    "realtime": ("flow.realtime", "stream generated audio live (press 'q' to quit)"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<11} {desc}")
        sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 1)
    module = importlib.import_module(COMMANDS[sys.argv[1]][0])
    sys.argv = [f"cli.py {sys.argv[1]}"] + sys.argv[2:]
    module.main()


if __name__ == "__main__":
    main()
