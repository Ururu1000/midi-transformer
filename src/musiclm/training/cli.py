"""CLI entry point for training: parses overrides and launches the trainer."""
from __future__ import annotations

import argparse
from pathlib import Path

from musiclm.config import DEFAULT_CHECKPOINT_PATH, TrainConfig
from musiclm.training.trainer import main as run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="musiclm-train",
        description="Train the composer-conditioned music transformer.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-step batch size (CUDA)")
    parser.add_argument(
        "--accum", type=int, default=None, help="Gradient accumulation steps"
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume-mode",
        choices=("none", "weights", "full"),
        default=None,
        help=(
            "none = from scratch; weights = load weights + counters only "
            "(default); full = restore optimizer/scheduler/scaler/RNG too"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint used by --resume-mode weights",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B entirely")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    defaults = TrainConfig()
    cfg = TrainConfig(
        batch_size=args.batch_size if args.batch_size is not None else defaults.batch_size,
        gradient_accumulation_steps=(
            args.accum if args.accum is not None else defaults.gradient_accumulation_steps
        ),
        num_epochs=args.epochs if args.epochs is not None else defaults.num_epochs,
        learning_rate=args.lr if args.lr is not None else defaults.learning_rate,
        dropout=args.dropout if args.dropout is not None else defaults.dropout,
        seed=args.seed if args.seed is not None else defaults.seed,
        resume_mode=args.resume_mode if args.resume_mode is not None else defaults.resume_mode,
        resume_checkpoint_path=args.resume_checkpoint,
        use_wandb=not args.no_wandb,
        wandb_mode=args.wandb_mode if args.wandb_mode is not None else defaults.wandb_mode,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
