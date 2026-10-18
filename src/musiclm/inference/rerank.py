"""Best-of-N reranking: keep the candidate whose statistics look most real.

Each candidate is cut into windows of ``window_notes`` notes and every window
is scored by its mean absolute z-score against real music by the same
composer (key consistency, pitch-class entropy, range, polyphony, density,
inter-onset interval, compression). A penalty for stuck stretches (trills or
octave pairs that run for seconds) beyond the composer's own rate is added on
top. Lower is better.

Reference statistics come from MAESTRO and are built once with
``musiclm-refstats``; without them only the stuck penalty is used.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pretty_midi
from miditok import REMI

from musiclm.config import (
    COMPOSER_MAPPING_PATH,
    MAESTRO_METADATA_PATH,
    RAW_MIDI_DIR,
    REFERENCE_STATS_PATH,
)
from musiclm.evaluation.metrics import (
    WINDOW_FEATURES,
    iter_pitched_notes,
    load_midi,
    note_window_features,
    stuck_fraction,
)
from musiclm.inference.priming import drop_primer

ALL_COMPOSERS = "__all__"
WINDOW_NOTES = 600
# Windows needed before a composer gets its own statistics instead of the
# all-composer pool.
MIN_COMPOSER_WINDOWS = 20
MAX_PIECES_PER_COMPOSER = 80
# Candidates shorter than this are almost always broken generations.
MIN_NOTES = 50
# Applied to stuck time beyond the composer's own (~3% for Chopin). At 2.0 a
# candidate stuck 20% of the time still beat loop-free ones on feature fit.
STUCK_WEIGHT = 6.0

logger = logging.getLogger(__name__)


def _windows(notes: Sequence[pretty_midi.Note], size: int) -> list[list[pretty_midi.Note]]:
    ordered = sorted(notes, key=lambda n: (n.start, n.pitch))
    full = [ordered[i : i + size] for i in range(0, len(ordered) - size + 1, size)]
    return full or [ordered]


def _summarize(rows: list[dict[str, float]], stuck: list[float]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    return {
        "windows": len(rows),
        "mean": {name: float(frame[name].mean()) for name in WINDOW_FEATURES},
        # Floor keeps a near-constant feature from dominating the z-score.
        "std": {name: max(float(frame[name].std()), 1e-6) for name in WINDOW_FEATURES},
        "stuck_fraction": float(np.mean(stuck)) if stuck else 0.0,
    }


def build_reference_stats(
    metadata_path: Path = MAESTRO_METADATA_PATH,
    midi_dir: Path = RAW_MIDI_DIR,
    mapping_path: Path = COMPOSER_MAPPING_PATH,
    window_notes: int = WINDOW_NOTES,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-composer feature means/stds over real MAESTRO windows."""
    with open(mapping_path) as fh:
        composer_to_group: dict[str, str] = json.load(fh)["composer_to_group"]
    metadata = pd.read_csv(metadata_path)

    pieces: dict[str, list[Path]] = defaultdict(list)
    for composer, filename in zip(metadata.canonical_composer, metadata.midi_filename):
        group = composer_to_group.get(composer)
        if group is not None and (midi_dir / filename).exists():
            pieces[group].append(midi_dir / filename)

    rng = random.Random(seed)
    per_group_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_group_stuck: dict[str, list[float]] = defaultdict(list)
    for group, paths in sorted(pieces.items()):
        rng.shuffle(paths)
        for path in paths[:MAX_PIECES_PER_COMPOSER]:
            midi = load_midi(path)
            if midi is None:
                continue
            notes = list(iter_pitched_notes(midi))
            if len(notes) < window_notes:
                continue
            for window in _windows(notes, window_notes):
                per_group_rows[group].append(note_window_features(window))
                per_group_stuck[group].append(stuck_fraction(window))
        logger.info("%s: %d windows", group, len(per_group_rows[group]))

    all_rows = [row for rows in per_group_rows.values() for row in rows]
    all_stuck = [s for values in per_group_stuck.values() for s in values]
    if not all_rows:
        raise ValueError(f"No MAESTRO windows found under {midi_dir}")
    composers = {
        group: _summarize(rows, per_group_stuck[group])
        for group, rows in per_group_rows.items()
        if len(rows) >= MIN_COMPOSER_WINDOWS
    }
    composers[ALL_COMPOSERS] = _summarize(all_rows, all_stuck)
    return {"window_notes": window_notes, "composers": composers}


def load_reference_stats(path: Path = REFERENCE_STATS_PATH) -> dict[str, Any] | None:
    if not path.exists():
        logger.warning(
            "No reference stats at %s; best-of-N ranks by stuck stretches only. "
            "Run `musiclm-refstats` to build them.",
            path,
        )
        return None
    with open(path) as fh:
        return json.load(fh)


def score_notes(
    notes: Sequence[pretty_midi.Note],
    composer: str,
    stats: dict[str, Any] | None,
) -> float:
    """Distance from real music by *composer*; lower is better."""
    if len(notes) < MIN_NOTES:
        return math.inf
    stuck = stuck_fraction(notes)
    if stats is None:
        return STUCK_WEIGHT * stuck

    composers = stats["composers"]
    reference = composers.get(composer, composers[ALL_COMPOSERS])
    penalty = STUCK_WEIGHT * max(0.0, stuck - reference["stuck_fraction"])
    window_scores = []
    for window in _windows(notes, int(stats["window_notes"])):
        features = note_window_features(window)
        window_scores.append(
            np.mean([
                abs(features[name] - reference["mean"][name]) / reference["std"][name]
                for name in WINDOW_FEATURES
            ])
        )
    return float(np.mean(window_scores)) + penalty


def score_candidates(
    tokenizer: REMI,
    candidates: Sequence[list[int]],
    composer: str,
    stats: dict[str, Any] | None,
    primer_len: int = 0,
) -> list[float]:
    """Score each token sequence; a primer is excluded from its own score."""
    from musiclm.inference.sampler import tokens_to_midi_file

    scores = []
    with tempfile.TemporaryDirectory() as tmp:
        for index, token_ids in enumerate(candidates):
            if primer_len:
                token_ids = drop_primer(token_ids, primer_len, tokenizer)
            path = Path(tmp) / f"candidate_{index}.mid"
            tokens_to_midi_file(tokenizer, token_ids, path)
            midi = load_midi(path)
            notes = list(iter_pitched_notes(midi)) if midi is not None else []
            scores.append(score_notes(notes, composer, stats))
    logger.info("Candidate scores (lower is better): %s", [round(s, 3) for s in scores])
    return scores


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="musiclm-refstats",
        description="Build per-composer reference statistics for best-of-N reranking.",
    )
    parser.add_argument("--output", type=Path, default=REFERENCE_STATS_PATH)
    parser.add_argument("--window-notes", type=int, default=WINDOW_NOTES)
    args = parser.parse_args(argv)

    stats = build_reference_stats(window_notes=args.window_notes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    logger.info(
        "Wrote %s (%d composer groups)", args.output, len(stats["composers"]) - 1
    )


if __name__ == "__main__":
    main()
