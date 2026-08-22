"""Dataset preprocessing: metadata, filtering, BPE training and tokenization."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pretty_midi
import torch
from miditok import REMI, TokSequence
from torch import Tensor

from musiclm.config import (
    COMPOSER_MAPPING_PATH,
    GIANTMIDI_DIR,
    MAESTRO_METADATA_PATH,
    PROCESSED_DIR,
    RAW_MIDI_DIR,
    TOKENIZER_PATH,
    TRAIN_TOKENS_PATH,
    UNIFIED_METADATA_PATH,
    VAL_TOKENS_PATH,
)
from musiclm.data.tokenizer import (
    OTHER_COMPOSER,
    RESERVED_COMPOSERS,
    TOP_COMPOSERS,
    UNCONDITIONAL_COMPOSER,
    ComposerREMI,
    build_tokenizer,
    composer_vocab_token,
    encode_base_ids_batch,
    learned_token_id,
)

MAX_SEQ_LEN = 4096
# Non-overlapping chunks: with GiantMIDI, 50% overlap mostly aids memorization.
CHUNK_STRIDE = MAX_SEQ_LEN - 1

# BPE target vocabulary. Merges frequent note/rhythm patterns into single ids,
# so a fixed 4096-token window covers substantially more music.
BPE_VOCAB_SIZE = 2048

# Pitch-shift augmentation range (semitones). BPE ids cannot be transposed
# directly (one id may span several notes), so every transposition is encoded
# at tokenization time and the training dataset picks a variant per document
# per sample.
MIN_PITCH_SHIFT = -6
MAX_PITCH_SHIFT = 5
TRAIN_SHIFTS = tuple(range(MIN_PITCH_SHIFT, MAX_PITCH_SHIFT + 1))
VAL_SHIFTS = (0,)

DOCS_FORMAT = "docs_v2_bpe"

# GiantMIDI transcription quality gates. Cap is soft enough for Romantic
# piano (pedal sustains) but still drops extreme transcription artifacts.
MAX_POLYPHONY = 16
MIN_NOTES_PER_SEC = 0.5
MAX_NOTES_PER_SEC = 30.0
GIANTMIDI_VAL_FRACTION = 0.05

# Map GiantMIDI "Last, First" (and common variants) onto MAESTRO canonical names
# so CFG composer tokens stay shared across sources.
GIANTMIDI_COMPOSER_ALIASES: dict[str, str] = {
    "Chopin, Frédéric": "Frédéric Chopin",
    "Chopin, Frederic": "Frédéric Chopin",
    "Bach, Johann Sebastian": "Johann Sebastian Bach",
    "Beethoven, Ludwig van": "Ludwig van Beethoven",
    "Liszt, Franz": "Franz Liszt",
    "Debussy, Claude": "Claude Debussy",
    "Rachmaninoff, Sergei": "Sergei Rachmaninoff",
    "Rachmaninov, Sergei": "Sergei Rachmaninoff",
    "Mozart, Wolfgang Amadeus": "Wolfgang Amadeus Mozart",
    "Schubert, Franz": "Franz Schubert",
    "Schumann, Robert": "Robert Schumann",
    "Brahms, Johannes": "Johannes Brahms",
    "Haydn, Joseph": "Joseph Haydn",
    "Haydn, Franz Joseph": "Joseph Haydn",
    "Scriabin, Alexander": "Alexander Scriabin",
    "Scarlatti, Domenico": "Domenico Scarlatti",
    "Mendelssohn, Felix": "Felix Mendelssohn",
    "Tchaikovsky, Pyotr Ilyich": "Pyotr Ilyich Tchaikovsky",
    "Tchaikovsky, Pyotr": "Pyotr Ilyich Tchaikovsky",
    "Handel, George Frideric": "George Frideric Handel",
    "Handel, Georg Friedrich": "George Frideric Handel",
    "Grieg, Edvard": "Edvard Grieg",
    "Franck, César": "César Franck",
    "Franck, Cesar": "César Franck",
    "Albéniz, Isaac": "Isaac Albéniz",
    "Albeniz, Isaac": "Isaac Albéniz",
    "Mussorgsky, Modest": "Modest Mussorgsky",
    "Clementi, Muzio": "Muzio Clementi",
    "Weber, Carl Maria von": "Carl Maria von Weber",
    "Berg, Alban": "Alban Berg",
    "Rameau, Jean-Philippe": "Jean-Philippe Rameau",
    "Purcell, Henry": "Henry Purcell",
    "Pachelbel, Johann": "Johann Pachelbel",
    "Janáček, Leoš": "Leoš Janáček",
    "Janacek, Leos": "Leoš Janáček",
    "Enescu, George": "George Enescu",
    "Soler, Antonio": "Antonio Soler",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MidiRecord:
    midi_path: Path
    composer: str
    split: str


def load_maestro_metadata(metadata_path: Path, dataset_dir: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {"midi_filename", "canonical_composer", "split"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metadata missing required columns: {missing}")

    metadata = metadata.loc[:, sorted(required_columns)].dropna().copy()
    metadata["midi_path"] = metadata["midi_filename"].map(
        lambda filename: dataset_dir / str(filename)
    )
    metadata["source"] = "maestro"
    return metadata


def parse_giantmidi_composer(filename: str) -> str | None:
    """Parse ``Surname, Firstname`` from GiantMIDI audio_name-style filenames.

    Filenames look like ``Last, First, Title[, more title], youtubeId.mid``.
    Composer is always the first two comma-separated fields (firstname may
    contain spaces, e.g. ``Johann Sebastian``); the last field is the YouTube id.
    """
    stem = Path(filename).name
    if not stem.lower().endswith(".mid"):
        return None
    parts = [part.strip() for part in stem[:-4].split(",")]
    if len(parts) < 3:
        return None
    return f"{parts[0]}, {parts[1]}"


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def giantmidi_to_maestro_composer(raw_composer: str) -> str:
    if raw_composer in GIANTMIDI_COMPOSER_ALIASES:
        return GIANTMIDI_COMPOSER_ALIASES[raw_composer]
    # Accent-insensitive alias lookup.
    key = _strip_accents(raw_composer).lower()
    for alias, target in GIANTMIDI_COMPOSER_ALIASES.items():
        if _strip_accents(alias).lower() == key:
            return target
    return raw_composer


def stable_giantmidi_split(
    filename: str, val_fraction: float = GIANTMIDI_VAL_FRACTION
) -> str:
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "validation" if bucket < val_fraction else "train"


def load_giantmidi_metadata(giantmidi_dir: Path) -> pd.DataFrame:
    if not giantmidi_dir.exists():
        raise FileNotFoundError(f"GiantMIDI directory not found: {giantmidi_dir}")

    midi_paths = sorted(giantmidi_dir.rglob("*.mid"))
    if not midi_paths:
        raise FileNotFoundError(f"No .mid files under {giantmidi_dir}")

    rows: list[dict[str, str]] = []
    parse_failures = 0
    for midi_path in midi_paths:
        relative = midi_path.relative_to(RAW_MIDI_DIR).as_posix()
        raw_composer = parse_giantmidi_composer(midi_path.name)
        if raw_composer is None:
            parse_failures += 1
            logger.warning("Skipping GiantMIDI file with unparseable name: %s", midi_path)
            continue
        composer = giantmidi_to_maestro_composer(raw_composer)
        split = stable_giantmidi_split(midi_path.name)
        rows.append(
            {
                "midi_filename": relative,
                "canonical_composer": composer,
                "split": split,
                "midi_path": str(midi_path),
                "source": "giantmidi",
            }
        )

    if not rows:
        raise ValueError("No GiantMIDI rows after composer parsing")

    metadata = pd.DataFrame(rows)
    metadata["midi_path"] = metadata["midi_path"].map(Path)
    logger.info(
        "GiantMIDI metadata: %d files | parse_failures=%d | train=%d | val=%d",
        len(metadata),
        parse_failures,
        int((metadata["split"] == "train").sum()),
        int((metadata["split"] == "validation").sum()),
    )
    return metadata


def build_unified_metadata(
    maestro_path: Path,
    dataset_dir: Path,
    giantmidi_dir: Path,
) -> pd.DataFrame:
    maestro = load_maestro_metadata(maestro_path, dataset_dir)
    if giantmidi_dir.exists() and any(giantmidi_dir.rglob("*.mid")):
        giant = load_giantmidi_metadata(giantmidi_dir)
        metadata = pd.concat([maestro, giant], ignore_index=True)
    else:
        logger.warning(
            "GiantMIDI dir missing or empty (%s); using MAESTRO only",
            giantmidi_dir,
        )
        metadata = maestro
    return metadata


def build_composer_mapping(metadata: pd.DataFrame) -> dict[str, str]:
    counts = metadata["canonical_composer"].astype(str).value_counts()
    reserved = {composer_vocab_token(name) for name in RESERVED_COMPOSERS}
    top_composers = {
        composer
        for composer in counts.head(TOP_COMPOSERS).index
        if composer_vocab_token(str(composer)) not in reserved
    }
    return {
        composer: composer if composer in top_composers else OTHER_COMPOSER
        for composer in counts.index
    }


def build_composer_groups(mapping: dict[str, str]) -> list[str]:
    groups = [
        composer for composer, group in mapping.items() if group != OTHER_COMPOSER
    ]
    groups.extend(RESERVED_COMPOSERS)
    return groups


def save_composer_mapping(mapping: dict[str, str], output_path: Path) -> None:
    payload = {
        "top_composers": [
            composer for composer, group in mapping.items() if group != OTHER_COMPOSER
        ],
        "other_token": composer_vocab_token(OTHER_COMPOSER),
        "unconditional_token": composer_vocab_token(UNCONDITIONAL_COMPOSER),
        "composer_to_group": mapping,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_pitch_shift_maps(
    tokenizer: REMI,
    min_shift: int = MIN_PITCH_SHIFT,
    max_shift: int = MAX_PITCH_SHIFT,
) -> dict[int, Tensor]:
    """Base-vocabulary id remap tensors, one per transposition."""
    pitch_token_ids = {
        int(token.split("_")[1]): token_id
        for token, token_id in tokenizer.vocab.items()
        if token.startswith("Pitch_")
    }
    assert pitch_token_ids, "No Pitch_* tokens found in tokenizer vocabulary"

    base_vocab_size = len(tokenizer.vocab)
    lowest, highest = min(pitch_token_ids), max(pitch_token_ids)
    shift_maps: dict[int, Tensor] = {}
    for shift in range(min_shift, max_shift + 1):
        # Identity mapping; only Pitch ids are redirected to the transposed pitch.
        mapping = torch.arange(base_vocab_size, dtype=torch.long)
        for pitch, token_id in pitch_token_ids.items():
            # Clamp so transposition never leaves the tokenizer's pitch range.
            shifted = min(max(pitch + shift, lowest), highest)
            mapping[token_id] = pitch_token_ids[shifted]
        shift_maps[shift] = mapping
    return shift_maps


def max_polyphony(midi: pretty_midi.PrettyMIDI) -> int:
    events: list[tuple[float, int]] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            events.append((note.start, 1))
            events.append((note.end, -1))
    if not events:
        return 0
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def validate_midi_file(midi_path: Path, *, apply_quality_filter: bool) -> bool:
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        logger.warning("Skipping invalid MIDI file %s: %s", midi_path, exc)
        return False

    note_count = sum(len(instrument.notes) for instrument in midi.instruments)
    if note_count == 0:
        logger.warning("Skipping MIDI file without notes: %s", midi_path)
        return False

    if not apply_quality_filter:
        return True

    duration = float(midi.get_end_time())
    if duration <= 0.0:
        logger.warning("Skipping MIDI with non-positive duration: %s", midi_path)
        return False

    density = note_count / duration
    if density < MIN_NOTES_PER_SEC or density > MAX_NOTES_PER_SEC:
        logger.debug(
            "Skipping MIDI for note density %.2f notes/s: %s",
            density,
            midi_path,
        )
        return False

    polyphony = max_polyphony(midi)
    if polyphony > MAX_POLYPHONY:
        logger.debug(
            "Skipping MIDI for max polyphony %d > %d: %s",
            polyphony,
            MAX_POLYPHONY,
            midi_path,
        )
        return False

    return True


def as_token_sequences(tokens: TokSequence | list[TokSequence]) -> list[TokSequence]:
    return tokens if isinstance(tokens, list) else [tokens]


def collect_split_records(
    metadata: pd.DataFrame,
    split: str,
) -> tuple[list[MidiRecord], int, int]:
    records: list[MidiRecord] = []
    missing_files = 0
    quality_rejects = 0
    split_rows = metadata[metadata["split"].astype(str).str.lower() == split]
    for row in split_rows.itertuples(index=False):
        midi_path = Path(row.midi_path)
        if not midi_path.exists():
            logger.warning("Skipping missing MIDI file: %s", midi_path)
            missing_files += 1
            continue
        source = str(getattr(row, "source", "maestro"))
        apply_quality = source == "giantmidi"
        if not validate_midi_file(midi_path, apply_quality_filter=apply_quality):
            if apply_quality:
                quality_rejects += 1
            continue
        records.append(
            MidiRecord(
                midi_path=midi_path,
                composer=str(row.canonical_composer),
                split=str(row.split),
            )
        )
    return records, missing_files, quality_rejects


def chunk_base_ids(ids: list[int], max_seq_len: int, stride: int) -> list[list[int]]:
    # One slot per chunk is reserved for the composer prefix added by the
    # training dataset.
    content_length = max_seq_len - 1
    assert 0 < stride <= content_length, (
        f"stride={stride} must be in range [1, {content_length}]"
    )
    chunks: list[list[int]] = []
    for start in range(0, len(ids), stride):
        content = ids[start : start + content_length]
        if content:
            chunks.append(content)
    return chunks


def tokenize_split(
    records: list[MidiRecord],
    tokenizer: ComposerREMI,
    composer_mapping: dict[str, str],
    max_seq_len: int,
    chunk_stride: int,
    shifts: tuple[int, ...],
) -> list[dict[str, Any]]:
    shift_maps = build_pitch_shift_maps(tokenizer)
    docs: list[dict[str, Any]] = []
    processed_files = 0

    for record in records:
        composer_group = composer_mapping.get(record.composer, OTHER_COMPOSER)
        composer_id = learned_token_id(
            tokenizer, composer_vocab_token(composer_group)
        )
        try:
            tokens = tokenizer.encode(record.midi_path, encode_ids=False)
        except Exception as exc:
            logger.warning(
                "Skipping MIDI file %s: tokenization failed: %s",
                record.midi_path,
                exc,
            )
            continue

        for sequence in as_token_sequences(tokens):
            base_ids = list(sequence.ids)
            if not base_ids:
                continue
            for chunk in chunk_base_ids(base_ids, max_seq_len, chunk_stride):
                chunk_tensor = torch.tensor(chunk, dtype=torch.long)
                shifted_batch = [
                    shift_maps[shift][chunk_tensor].tolist() for shift in shifts
                ]
                encoded_batch = encode_base_ids_batch(tokenizer, shifted_batch)
                variants = {
                    shift: torch.tensor(encoded, dtype=torch.int32)
                    for shift, encoded in zip(shifts, encoded_batch)
                }
                docs.append({"composer_id": composer_id, "variants": variants})
        processed_files += 1

    if not docs:
        raise ValueError("No token sequences were created for this split")

    logger.info("Tokenized %d files into %d documents", processed_files, len(docs))
    return docs


def save_tokenized_docs(
    docs: list[dict[str, Any]],
    tokenizer: ComposerREMI,
    shifts: tuple[int, ...],
    output_path: Path,
) -> None:
    torch.save(
        {
            "format": DOCS_FORMAT,
            "docs": docs,
            "shifts": list(shifts),
            "pad_token_id": learned_token_id(tokenizer, "PAD_None"),
            "vocab_size": len(tokenizer),
        },
        output_path,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metadata = build_unified_metadata(MAESTRO_METADATA_PATH, RAW_MIDI_DIR, GIANTMIDI_DIR)
    metadata.to_csv(UNIFIED_METADATA_PATH, index=False)
    composer_mapping = build_composer_mapping(metadata)
    composer_groups = build_composer_groups(composer_mapping)

    source_counts = metadata["source"].value_counts().to_dict()
    logger.info(
        "Loaded %d metadata rows | sources=%s | composers=%d | conditioned groups=%d",
        len(metadata),
        source_counts,
        len(composer_mapping),
        len(composer_groups),
    )

    train_records, train_missing, train_rejects = collect_split_records(
        metadata, "train"
    )
    val_records, val_missing, val_rejects = collect_split_records(
        metadata, "validation"
    )
    logger.info(
        "Valid MIDI files: train=%d (missing %d, quality_reject %d) | "
        "validation=%d (missing %d, quality_reject %d)",
        len(train_records),
        train_missing,
        train_rejects,
        len(val_records),
        val_missing,
        val_rejects,
    )

    tokenizer = build_tokenizer(composer_groups)
    logger.info(
        "Training BPE: base vocab %d -> target %d on %d files",
        len(tokenizer.vocab),
        BPE_VOCAB_SIZE,
        len(train_records),
    )
    tokenizer.train(
        vocab_size=BPE_VOCAB_SIZE,
        model="BPE",
        files_paths=[str(record.midi_path) for record in train_records],
    )
    assert tokenizer.is_trained, "BPE training did not mark the tokenizer trained"
    # The learned BPE model is serialized inside the same params file, so a
    # plain ComposerREMI(params=...) restores identical encodings.
    tokenizer.save(TOKENIZER_PATH)
    save_composer_mapping(composer_mapping, COMPOSER_MAPPING_PATH)

    train_docs = tokenize_split(
        train_records,
        tokenizer,
        composer_mapping,
        MAX_SEQ_LEN,
        CHUNK_STRIDE,
        TRAIN_SHIFTS,
    )
    val_docs = tokenize_split(
        val_records,
        tokenizer,
        composer_mapping,
        MAX_SEQ_LEN,
        CHUNK_STRIDE,
        VAL_SHIFTS,
    )

    save_tokenized_docs(train_docs, tokenizer, TRAIN_SHIFTS, TRAIN_TOKENS_PATH)
    save_tokenized_docs(val_docs, tokenizer, VAL_SHIFTS, VAL_TOKENS_PATH)

    encoded_lengths = sum(int(doc["variants"][0].numel()) for doc in train_docs)
    logger.info(
        "Saved train=%d docs to %s | validation=%d docs to %s | tokenizer=%s "
        "| learned vocab=%d | train tokens after BPE=%d",
        len(train_docs),
        TRAIN_TOKENS_PATH,
        len(val_docs),
        VAL_TOKENS_PATH,
        TOKENIZER_PATH,
        len(tokenizer),
        encoded_lengths,
    )


if __name__ == "__main__":
    main()
