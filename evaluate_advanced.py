"""Advanced statistical evaluation of generated MIDI against a validation corpus.

Implements six metric families:
  1. Scale Consistency   – Krumhansl-Schmuckler key detection + in-key note rate
  2. Pitch Class Entropy – Shannon entropy of the 12-bin pitch class histogram
  3. Pitch Range         – semitone span between lowest and highest pitch
  4. Polyphony Rate & Note Density with KL Divergence across corpora
  5. Groove Consistency  – Inter-Onset Interval (IOI) histogram analysis
  6. Compression Ratio / Structure – LZ77 (zlib) compression of a note string

Usage:
    python evaluate_advanced.py data/generated data/raw_midi/validation
    python evaluate_advanced.py gen-dir data/generated val-dir data/raw_midi
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
from scipy.stats import entropy as scipy_entropy


# Constants


PITCH_CLASSES = 12
# Laplace smoothing prevents infinite KL when a bin is empty.
KL_EPSILON = 1e-8
# Number of equal-width bins for IOI and polyphony/density KLD histograms.
IOI_BINS = 50
DENSITY_KLD_BINS = 30
# Cap on IOI values (seconds) to ignore pathological long rests.
IOI_CAP_SECONDS = 4.0

logger = logging.getLogger(__name__)


# Krumhansl-Schmuckler key profiles


# Probe-tone ratings from Krumhansl & Kessler (1982).
# Index 0 = tonic, chromatically ascending.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

# Pitch-class names for labelling keys (C=0 .. B=11).
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# The diatonic pitch classes (semitone offsets from the tonic) for major/minor.
_MAJOR_SCALE_STEPS = frozenset({0, 2, 4, 5, 7, 9, 11})
_MINOR_SCALE_STEPS = frozenset({0, 2, 3, 5, 7, 8, 10})  # natural minor


def _rotate(array: np.ndarray, shift: int) -> np.ndarray:
    """Circular rotation of a 1-D array to the right by *shift* positions."""
    return np.roll(array, -shift)


def detect_key(pch: np.ndarray) -> tuple[str, str]:
    """Return ``(root_name, mode)`` using the Krumhansl-Schmuckler algorithm.

    *pch* is a 12-element pitch-class histogram (need not be normalised).
    """
    best_corr = -2.0
    best_root = 0
    best_mode = "major"

    for root in range(PITCH_CLASSES):
        rotated = _rotate(pch, root)
        for profile, mode in ((_MAJOR_PROFILE, "major"), (_MINOR_PROFILE, "minor")):
            corr = float(np.corrcoef(rotated, profile)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_root = root
                best_mode = mode

    return _PC_NAMES[best_root], best_mode


def scale_consistency(
    pitches: Sequence[int],
    root_name: str,
    mode: str,
) -> float:
    """Fraction of *pitches* that belong to the detected key's diatonic set."""
    if not pitches:
        return 0.0

    root_pc = _PC_NAMES.index(root_name)
    steps = _MAJOR_SCALE_STEPS if mode == "major" else _MINOR_SCALE_STEPS
    in_key = sum(1 for p in pitches if (p - root_pc) % PITCH_CLASSES in steps)
    return in_key / len(pitches)



# Low-level MIDI helpers (mirrors existing evaluate.py conventions)



def load_midi(midi_path: Path) -> pretty_midi.PrettyMIDI | None:
    """Attempt to parse a MIDI file, returning ``None`` on failure."""
    try:
        return pretty_midi.PrettyMIDI(str(midi_path))
    except (OSError, ValueError, KeyError, EOFError, IndexError, Exception) as exc:
        logger.warning("Skipping unreadable MIDI %s: %s", midi_path.name, exc)
        return None


def iter_pitched_notes(
    midi: pretty_midi.PrettyMIDI,
) -> Iterator[pretty_midi.Note]:
    """Yield every non-drum note across all instruments."""
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        yield from instrument.notes



# Per-file metric extraction



@dataclass(frozen=True)
class FileMetrics:
    """Container for every metric extracted from one MIDI file."""

    path: Path
    # 1. Scale consistency
    detected_key: str  # e.g. "C major"
    scale_consistency: float
    # 2. Pitch-class entropy
    pc_entropy: float
    # 3. Pitch range
    pitch_range: int
    # 4. Polyphony rate and note density
    polyphony_rate: float  # mean concurrent notes
    note_density: float    # notes per second
    # 5. Groove – raw IOI values stored for corpus-level histogramming
    ioi_mean: float
    ioi_std: float
    # 6. Compression ratio (lower → more repetitive internal structure)
    compression_ratio: float
    # Bookkeeping
    note_count: int
    duration: float
    # Raw pitch-class histogram (12-bin, normalised)
    pch: np.ndarray = field(repr=False)


def _pitch_class_histogram(notes: list[pretty_midi.Note]) -> np.ndarray:
    """Duration×velocity-weighted 12-bin pitch-class histogram, normalised."""
    histogram = np.zeros(PITCH_CLASSES, dtype=np.float64)
    for note in notes:
        weight = (note.end - note.start) * (note.velocity / 127.0)
        histogram[note.pitch % PITCH_CLASSES] += weight
    total = histogram.sum()
    return histogram / total if total > 0.0 else histogram


def _active_duration(notes: list[pretty_midi.Note]) -> float:
    """Seconds between the first note onset and the last note offset."""
    if not notes:
        return 0.0
    first = min(n.start for n in notes)
    last = max(n.end for n in notes)
    span = last - first
    return span if span > 0.0 else 0.0


def _polyphony_rate(notes: list[pretty_midi.Note], duration: float) -> float:
    """Average number of concurrently sounding notes.

    Computed as sum-of-note-durations / total-active-duration.  This equals
    the expected number of notes active at a uniformly sampled instant.
    """
    if duration <= 0.0:
        return 0.0
    total_sounding = sum(n.end - n.start for n in notes)
    return total_sounding / duration


def _inter_onset_intervals(notes: list[pretty_midi.Note]) -> np.ndarray:
    """Sorted onset differences, capped at IOI_CAP_SECONDS."""
    if len(notes) < 2:
        return np.array([], dtype=np.float64)
    onsets = np.array(sorted({n.start for n in notes}), dtype=np.float64)
    ioi = np.diff(onsets)
    return ioi[ioi <= IOI_CAP_SECONDS]


def _compression_ratio(notes: list[pretty_midi.Note]) -> float:
    """LZ77 compression ratio of a pitch-duration string representation.

    A ratio close to 1.0 means the piece is nearly incompressible (high
    randomness / low repetition); lower values indicate more structural
    repetition.
    """
    if not notes:
        return 1.0
    # Quantise durations to 50 ms grid to surface rhythmic patterns.
    tokens = [
        f"{n.pitch}:{round((n.end - n.start) / 0.05)}"
        for n in sorted(notes, key=lambda n: (n.start, n.pitch))
    ]
    raw = ",".join(tokens).encode("ascii")
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw) if len(raw) > 0 else 1.0


def extract_file_metrics(midi_path: Path) -> FileMetrics | None:
    """Compute every metric for one MIDI file."""
    midi = load_midi(midi_path)
    if midi is None:
        return None

    notes = list(iter_pitched_notes(midi))
    if not notes:
        logger.warning("Skipping MIDI without pitched notes: %s", midi_path.name)
        return None

    pitches = [n.pitch for n in notes]
    pch = _pitch_class_histogram(notes)
    duration = _active_duration(notes)

    root, mode = detect_key(pch)
    sc = scale_consistency(pitches, root, mode)

    # Shannon entropy (log base 2 → bits; log base e → nats).  We use nats
    # for consistency with scipy and the existing evaluate.py KL divergences.
    pc_ent = float(scipy_entropy(pch + KL_EPSILON))

    pr = max(pitches) - min(pitches)
    poly = _polyphony_rate(notes, duration)
    density = len(notes) / duration if duration > 0.0 else 0.0

    ioi = _inter_onset_intervals(notes)
    ioi_mean = float(ioi.mean()) if ioi.size > 0 else 0.0
    ioi_std = float(ioi.std()) if ioi.size > 0 else 0.0

    cr = _compression_ratio(notes)

    return FileMetrics(
        path=midi_path,
        detected_key=f"{root} {mode}",
        scale_consistency=sc,
        pc_entropy=pc_ent,
        pitch_range=pr,
        polyphony_rate=poly,
        note_density=density,
        ioi_mean=ioi_mean,
        ioi_std=ioi_std,
        compression_ratio=cr,
        note_count=len(notes),
        duration=duration,
        pch=pch,
    )



# Corpus-level aggregation



@dataclass
class CorpusMetrics:
    """Aggregated statistics for one corpus (generated or validation)."""

    name: str
    file_count: int
    total_notes: int

    # 1. Scale consistency
    mean_scale_consistency: float
    # 2. Pitch-class entropy
    mean_pc_entropy: float
    # 3. Pitch range
    mean_pitch_range: float
    median_pitch_range: float
    # 4. Polyphony & density
    mean_polyphony_rate: float
    mean_note_density: float
    # 5. Groove
    mean_ioi_mean: float
    mean_ioi_std: float
    # 6. Structure
    mean_compression_ratio: float

    # Raw per-file arrays kept for KLD computation between corpora.
    densities: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    polyphonies: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    mean_pch: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros(PITCH_CLASSES)
    )
    # All IOI values concatenated across files for histogram KLD.
    all_iois: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def collect_corpus_metrics(
    name: str,
    midi_paths: list[Path],
) -> CorpusMetrics:
    """Process every MIDI path and aggregate per-file results."""
    per_file = [
        metrics
        for path in midi_paths
        if (metrics := extract_file_metrics(path)) is not None
    ]
    if not per_file:
        raise ValueError(f"No usable MIDI files in corpus '{name}'")

    densities = np.array([m.note_density for m in per_file])
    polyphonies = np.array([m.polyphony_rate for m in per_file])
    pitch_ranges = np.array([m.pitch_range for m in per_file])
    sc_values = np.array([m.scale_consistency for m in per_file])
    pc_entropies = np.array([m.pc_entropy for m in per_file])
    ioi_means = np.array([m.ioi_mean for m in per_file])
    ioi_stds = np.array([m.ioi_std for m in per_file])
    comp_ratios = np.array([m.compression_ratio for m in per_file])
    pch_matrix = np.stack([m.pch for m in per_file])

    # Gather all raw IOI values across the corpus for histogram KLD.
    all_iois_list: list[np.ndarray] = []
    for path in midi_paths:
        midi = load_midi(path)
        if midi is None:
            continue
        notes = list(iter_pitched_notes(midi))
        ioi = _inter_onset_intervals(notes)
        if ioi.size > 0:
            all_iois_list.append(ioi)
    all_iois = np.concatenate(all_iois_list) if all_iois_list else np.array([])

    logger.info(
        "Corpus %-12s | files=%d/%d | notes=%d | density=%.2f n/s | "
        "polyphony=%.2f | scale_consistency=%.1f%%",
        name,
        len(per_file),
        len(midi_paths),
        sum(m.note_count for m in per_file),
        float(densities.mean()),
        float(polyphonies.mean()),
        float(sc_values.mean()) * 100,
    )

    return CorpusMetrics(
        name=name,
        file_count=len(per_file),
        total_notes=sum(m.note_count for m in per_file),
        mean_scale_consistency=float(sc_values.mean()),
        mean_pc_entropy=float(pc_entropies.mean()),
        mean_pitch_range=float(pitch_ranges.mean()),
        median_pitch_range=float(np.median(pitch_ranges)),
        mean_polyphony_rate=float(polyphonies.mean()),
        mean_note_density=float(densities.mean()),
        mean_ioi_mean=float(ioi_means.mean()),
        mean_ioi_std=float(ioi_stds.mean()),
        mean_compression_ratio=float(comp_ratios.mean()),
        densities=densities,
        polyphonies=polyphonies,
        mean_pch=pch_matrix.mean(axis=0),
        all_iois=all_iois,
    )



# Cross-corpus KL divergence helpers



def _histogram_kld(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    bins: int,
    label: str,
) -> float:
    """KL(P || Q) between two sets of continuous samples via histogram binning.

    Both sample arrays are binned on the *same* edges (union range) so the
    supports are aligned.
    """
    if samples_p.size == 0 or samples_q.size == 0:
        logger.warning("Empty sample array for %s KLD; returning NaN.", label)
        return float("nan")

    lo = min(samples_p.min(), samples_q.min())
    hi = max(samples_p.max(), samples_q.max())
    if lo == hi:
        return 0.0

    edges = np.linspace(lo, hi, bins + 1)
    p_hist, _ = np.histogram(samples_p, bins=edges, density=False)
    q_hist, _ = np.histogram(samples_q, bins=edges, density=False)

    # Normalise to probability distributions with Laplace smoothing.
    p = (p_hist.astype(np.float64) + KL_EPSILON)
    q = (q_hist.astype(np.float64) + KL_EPSILON)
    p /= p.sum()
    q /= q.sum()

    return float(scipy_entropy(p, q))


def pch_kl_divergence(gen_pch: np.ndarray, ref_pch: np.ndarray) -> float:
    """KL divergence between two 12-bin pitch-class distributions."""
    p = gen_pch.astype(np.float64) + KL_EPSILON
    q = ref_pch.astype(np.float64) + KL_EPSILON
    p /= p.sum()
    q /= q.sum()
    return float(scipy_entropy(p, q))



# Summary table



def build_summary(
    gen: CorpusMetrics,
    val: CorpusMetrics,
) -> pd.DataFrame:
    """Construct a comparative pandas DataFrame of every metric."""

    # Cross-corpus KL divergences.
    pch_kld = pch_kl_divergence(gen.mean_pch, val.mean_pch)
    density_kld = _histogram_kld(
        gen.densities, val.densities, DENSITY_KLD_BINS, "note_density"
    )
    polyphony_kld = _histogram_kld(
        gen.polyphonies, val.polyphonies, DENSITY_KLD_BINS, "polyphony_rate"
    )
    ioi_kld = _histogram_kld(gen.all_iois, val.all_iois, IOI_BINS, "IOI")

    rows: list[dict[str, object]] = [
        #  Bookkeeping 
        {"metric": "files_processed", "generated": gen.file_count,
         "validation": val.file_count, "KLD": ""},
        {"metric": "total_notes", "generated": gen.total_notes,
         "validation": val.total_notes, "KLD": ""},
        #  1. Scale consistency 
        {"metric": "scale_consistency (%)",
         "generated": round(gen.mean_scale_consistency * 100, 2),
         "validation": round(val.mean_scale_consistency * 100, 2),
         "KLD": ""},
        #  2. Pitch-class entropy 
        {"metric": "pc_entropy (nats)",
         "generated": round(gen.mean_pc_entropy, 4),
         "validation": round(val.mean_pc_entropy, 4),
         "KLD": round(pch_kld, 4)},
        #  3. Pitch range 
        {"metric": "pitch_range_mean (semitones)",
         "generated": round(gen.mean_pitch_range, 1),
         "validation": round(val.mean_pitch_range, 1),
         "KLD": ""},
        {"metric": "pitch_range_median (semitones)",
         "generated": round(gen.median_pitch_range, 1),
         "validation": round(val.median_pitch_range, 1),
         "KLD": ""},
        #  4. Polyphony & density 
        {"metric": "polyphony_rate (avg concurrent)",
         "generated": round(gen.mean_polyphony_rate, 3),
         "validation": round(val.mean_polyphony_rate, 3),
         "KLD": round(polyphony_kld, 4) if not math.isnan(polyphony_kld) else "N/A"},
        {"metric": "note_density (notes/s)",
         "generated": round(gen.mean_note_density, 3),
         "validation": round(val.mean_note_density, 3),
         "KLD": round(density_kld, 4) if not math.isnan(density_kld) else "N/A"},
        #  5. Groove consistency 
        {"metric": "ioi_mean (s)",
         "generated": round(gen.mean_ioi_mean, 4),
         "validation": round(val.mean_ioi_mean, 4),
         "KLD": round(ioi_kld, 4) if not math.isnan(ioi_kld) else "N/A"},
        {"metric": "ioi_std (s)",
         "generated": round(gen.mean_ioi_std, 4),
         "validation": round(val.mean_ioi_std, 4),
         "KLD": ""},
        #  6. Structural compression 
        {"metric": "compression_ratio (lower=more repetition)",
         "generated": round(gen.mean_compression_ratio, 4),
         "validation": round(val.mean_compression_ratio, 4),
         "KLD": ""},
        #  Combined PCH KLD 
        {"metric": "pch_kl_divergence (nats)",
         "generated": round(pch_kld, 4),
         "validation": 0.0,
         "KLD": round(pch_kld, 4)},
    ]

    # Per-pitch-class breakdown.
    for i, name in enumerate(_PC_NAMES):
        rows.append({
            "metric": f"pch_{name}",
            "generated": round(float(gen.mean_pch[i]), 4),
            "validation": round(float(val.mean_pch[i]), 4),
            "KLD": "",
        })

    return pd.DataFrame(rows).set_index("metric")



# CLI



def _discover_midi(directory: Path) -> list[Path]:
    """Return sorted list of ``*.mid`` and ``*.midi`` files, recursively."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    paths = sorted(directory.rglob("*.mid")) + sorted(directory.rglob("*.midi"))
    # Deduplicate (a file may match both globs if named exactly `.midi`).
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)

    if not unique:
        raise FileNotFoundError(f"No .mid/.midi files found under {directory}")

    return unique


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare generated MIDI against a validation corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate_advanced.py data/generated data/raw_midi/validation\n"
            "  python evaluate_advanced.py gen-dir data/generated val-dir data/raw_midi\n"
        ),
    )
    parser.add_argument(
        "positional_dirs",
        nargs="*",
        metavar="DIR",
        help="Positional: generated_dir followed by validation_dir.",
    )
    parser.add_argument("gen-dir", type=Path, help="Path to generated MIDI files.")
    parser.add_argument("val-dir", type=Path, help="Path to validation MIDI files.")
    parser.add_argument(
        "csv",
        type=Path,
        default=None,
        help="Write the summary table to a CSV file.",
    )

    args = parser.parse_args(argv)

    # Resolve positional vs named arguments.
    if args.gen_dir and args.val_dir:
        pass  # named flags take priority
    elif len(args.positional_dirs) == 2:
        args.gen_dir = Path(args.positional_dirs[0])
        args.val_dir = Path(args.positional_dirs[1])
    else:
        parser.error(
            "Provide two positional directories (generated, validation) "
            "or use gen-dir and val-dir."
        )

    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args(argv)

    gen_paths = _discover_midi(args.gen_dir)
    val_paths = _discover_midi(args.val_dir)

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
