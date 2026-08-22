"""CLI for the statistical MIDI evaluation: compare two corpora."""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from musiclm.evaluation.metrics import build_summary, collect_corpus_metrics, discover_midi

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="musiclm-eval",
        description="Compare generated MIDI against a validation corpus.",
        epilog="Example: musiclm-eval data/generated data/raw_midi/validation --csv summary.csv",
    )
    parser.add_argument("gen_dir", type=Path, help="Directory of generated MIDI files.")
    parser.add_argument("val_dir", type=Path, help="Directory of validation MIDI files.")
    parser.add_argument("--csv", type=Path, default=None, help="Write summary table to CSV.")

    args = parser.parse_args(argv)

    gen_paths = discover_midi(args.gen_dir)
    val_paths = discover_midi(args.val_dir)

    logger.info("Generated corpus: %d files from %s", len(gen_paths), args.gen_dir)
    logger.info("Validation corpus: %d files from %s", len(val_paths), args.val_dir)

    gen_metrics = collect_corpus_metrics("generated", gen_paths)
    val_metrics = collect_corpus_metrics("validation", val_paths)

    summary = build_summary(gen_metrics, val_metrics)

    print("\n" + "=" * 80)
    print("  ADVANCED MIDI EVALUATION — COMPARATIVE SUMMARY")
    print("=" * 80)
    print(summary.to_string())
    print("=" * 80 + "\n")

    if args.csv:
        summary.to_csv(args.csv)
        logger.info("Summary written to %s", args.csv)


if __name__ == "__main__":
    main()
