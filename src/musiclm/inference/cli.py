"""CLI entry point for single-file generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from musiclm.config import GenerateConfig
from musiclm.inference.sampler import main as run_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="musiclm-generate",
        description="Generate a composer-conditioned MIDI file.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--composer", type=str, default=None, help='e.g. "Frédéric Chopin"'
    )
    parser.add_argument("--length", type=int, default=None, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument(
        "--cfg-scale", type=float, default=None, help="1.0 disables guidance"
    )
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--penalty-window", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="Set for reproducible sampling"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    defaults = GenerateConfig()
    cfg = GenerateConfig(
        checkpoint_path=args.checkpoint or defaults.checkpoint_path,
        tokenizer_path=args.tokenizer or defaults.tokenizer_path,
        output_path=args.output or defaults.output_path,
        composer=args.composer if args.composer is not None else defaults.composer,
        length=args.length if args.length is not None else defaults.length,
        temperature=(
            args.temperature if args.temperature is not None else defaults.temperature
        ),
        min_p=args.min_p if args.min_p is not None else defaults.min_p,
        cfg_scale=args.cfg_scale if args.cfg_scale is not None else defaults.cfg_scale,
        repetition_penalty=(
            args.repetition_penalty
            if args.repetition_penalty is not None
            else defaults.repetition_penalty
        ),
        penalty_window=(
            args.penalty_window if args.penalty_window is not None
            else defaults.penalty_window
        ),
        seed=args.seed,
    )
    run_generation(cfg)


if __name__ == "__main__":
    main()
