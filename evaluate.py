from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from scipy.stats import entropy

DATASET_DIR = Path("data/raw_midi")
METADATA_PATH = DATASET_DIR / "maestro-v3.0.0.csv"
GENERATED_DIR = Path("data/generated")

VALIDATION_SAMPLE_SIZE = 100
PITCH_CLASSES = 12
# Smoothing keeps KL divergence finite when a pitch class is absent in a set.
PCH_EPSILON = 1e-8
SEED = 42

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MidiStats:
    path: Path
    pch: np.ndarray
    note_density: float
    note_count: int
    duration: float


@dataclass(frozen=True)
class CorpusStats:
    name: str
    mean_pch: np.ndarray
    mean_note_density: float
    file_count: int
    total_notes: int


def load_midi(midi_path: Path) -> pretty_midi.PrettyMIDI | None:
    try:
        return pretty_midi.PrettyMIDI(str(midi_path))
    except (OSError, ValueError, KeyError, EOFError, IndexError) as exc:
        logger.warning("Skipping unreadable MIDI %s: %s", midi_path, exc)
        return None


def iter_pitched_notes(midi: pretty_midi.PrettyMIDI) -> Iterator[pretty_midi.Note]:
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        yield from instrument.notes


def extract_pch(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    # Weighting by duration x velocity measures tonal presence, not note events:
    # a held fortissimo chord tone dominates a grace note, as it does perceptually.
    histogram = np.zeros(PITCH_CLASSES, dtype=np.float64)
    for note in iter_pitched_notes(midi):
        weight = (note.end - note.start) * (note.velocity / 127.0)
        histogram[note.pitch % PITCH_CLASSES] += weight

    total = histogram.sum()
    if total == 0.0:
        return histogram

    normalized = histogram / total
    assert normalized.shape == (PITCH_CLASSES,), f"Got {normalized.shape}"
    return normalized


def count_notes(midi: pretty_midi.PrettyMIDI) -> int:
    return sum(1 for _ in iter_pitched_notes(midi))


def extract_active_duration(midi: pretty_midi.PrettyMIDI) -> float:
    # get_end_time() includes trailing silence and meta events, which deflates
    # density; span between first and last sounding note is the real playing time.
    starts: list[float] = []
    ends: list[float] = []
    for note in iter_pitched_notes(midi):
        starts.append(float(note.start))
        ends.append(float(note.end))

    if not starts:
        return 0.0

    active_duration = max(ends) - min(starts)
    return active_duration if active_duration > 0.0 else 0.0


def extract_note_density(midi: pretty_midi.PrettyMIDI) -> float:
    active_duration = extract_active_duration(midi)
    if active_duration <= 0.0:
        return 0.0
    return count_notes(midi) / active_duration


def extract_stats(midi_path: Path) -> MidiStats | None:
    midi = load_midi(midi_path)
    if midi is None:
        return None

    note_count = count_notes(midi)
    if note_count == 0:
        logger.warning("Skipping MIDI without notes: %s", midi_path)
        return None

    return MidiStats(
        path=midi_path,
        pch=extract_pch(midi),
        note_density=extract_note_density(midi),
        note_count=note_count,
        duration=extract_active_duration(midi),
    )


def find_validation_files(
    metadata_path: Path,
    dataset_dir: Path,
    sample_size: int,
    seed: int = SEED,
) -> list[Path]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {"split", "midi_filename"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metadata missing required columns: {missing}")

    validation_rows = metadata[metadata["split"].astype(str).str.lower() == "validation"]
    paths = [
        dataset_dir / str(filename)
        for filename in validation_rows["midi_filename"].dropna()
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No validation MIDI files found under {dataset_dir}")

    if len(existing) <= sample_size:
        return existing

    rng = random.Random(seed)
    return rng.sample(existing, sample_size)


def find_generated_files(generated_dir: Path) -> list[Path]:
    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    paths = sorted(generated_dir.glob("*.mid")) + sorted(generated_dir.glob("*.midi"))
    if not paths:
        raise FileNotFoundError(f"No MIDI files found in {generated_dir}")
    return paths


def collect_corpus_stats(name: str, midi_paths: list[Path]) -> CorpusStats:
    per_file = [stats for path in midi_paths if (stats := extract_stats(path))]
    if not per_file:
        raise ValueError(f"No usable MIDI files in corpus '{name}'")

    pch_matrix = np.stack([stats.pch for stats in per_file])
    assert pch_matrix.shape == (len(per_file), PITCH_CLASSES), f"Got {pch_matrix.shape}"

    mean_pch = pch_matrix.mean(axis=0)
    densities = np.array([stats.note_density for stats in per_file], dtype=np.float64)

    logger.info(
        "Corpus %s | files=%d/%d | mean note density=%.2f notes/s",
        name,
        len(per_file),
        len(midi_paths),
        float(densities.mean()),
    )
    return CorpusStats(
        name=name,
        mean_pch=mean_pch,
        mean_note_density=float(densities.mean()),
        file_count=len(per_file),
        total_notes=sum(stats.note_count for stats in per_file),
    )


def normalize_distribution(distribution: np.ndarray) -> np.ndarray:
    assert distribution.shape == (PITCH_CLASSES,), f"Got {distribution.shape}"
    smoothed = distribution.astype(np.float64) + PCH_EPSILON
    return smoothed / smoothed.sum()


def pch_kl_divergence(generated_pch: np.ndarray, reference_pch: np.ndarray) -> float:
    return float(
        entropy(
            normalize_distribution(generated_pch),
            normalize_distribution(reference_pch),
        )
    )


def build_summary_table(
    generated: CorpusStats,
    reference: CorpusStats,
    kl_divergence: float,
) -> pd.DataFrame:
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    rows: list[dict[str, object]] = [
        {
            "metric": "files",
            "generated": generated.file_count,
            "ground_truth": reference.file_count,
        },
        {
            "metric": "total_notes",
            "generated": generated.total_notes,
            "ground_truth": reference.total_notes,
        },
        {
            "metric": "note_density (notes/s)",
            "generated": round(generated.mean_note_density, 3),
            "ground_truth": round(reference.mean_note_density, 3),
        },
        {
            "metric": "pch_kl_divergence",
            "generated": round(kl_divergence, 4),
            "ground_truth": 0.0,
        },
    ]
    rows.extend(
        {
            "metric": f"pch_{name}",
            "generated": round(float(generated.mean_pch[index]), 4),
            "ground_truth": round(float(reference.mean_pch[index]), 4),
        }
        for index, name in enumerate(pitch_names)
    )
    return pd.DataFrame(rows).set_index("metric")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    validation_files = find_validation_files(
        METADATA_PATH,
        DATASET_DIR,
        VALIDATION_SAMPLE_SIZE,
    )
    generated_files = find_generated_files(GENERATED_DIR)

    reference = collect_corpus_stats("ground_truth", validation_files)
    generated = collect_corpus_stats("generated", generated_files)

    kl_divergence = pch_kl_divergence(generated.mean_pch, reference.mean_pch)
    density_ratio = generated.mean_note_density / max(reference.mean_note_density, 1e-8)

    summary = build_summary_table(generated, reference, kl_divergence)
    logger.info("Evaluation summary:\n%s", summary.to_string())
    logger.info(
        "PCH KL divergence (generated || ground truth): %.4f nats",
        kl_divergence,
    )
    logger.info(
        "Note density: generated %.2f vs ground truth %.2f notes/s (ratio %.2f)",
        generated.mean_note_density,
        reference.mean_note_density,
        density_ratio,
    )


if __name__ == "__main__":
    main()
