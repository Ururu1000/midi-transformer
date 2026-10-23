"""CLI entry point for training: parses overrides and launches the trainer."""
from __future__ import annotations

import argparse
from pathlib import Path

from musiclm.config import TrainConfig, with_overrides
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
        default=None,
        help="Checkpoint used by --resume-mode weights (default: checkpoints/model_best.pt)",
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B entirely")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    cfg = with_overrides(
        TrainConfig(),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        dropout=args.dropout,
        seed=args.seed,
        resume_mode=args.resume_mode,
        resume_checkpoint_path=args.resume_checkpoint,
        use_wandb=False if args.no_wandb else None,
        wandb_mode=args.wandb_mode,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
