from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pretty_midi
import torch
from miditok import REMI, TokSequence, TokenizerConfig
from torch import Tensor

MAX_SEQ_LEN = 4096
CHUNK_STRIDE = 2048
DATASET_DIR = Path("data/raw_midi")
METADATA_PATH = DATASET_DIR / "maestro-v3.0.0.csv"
PROCESSED_DIR = Path("data/processed")
TOKENIZER_PATH = PROCESSED_DIR / "tokenizer.json"
COMPOSER_MAPPING_PATH = PROCESSED_DIR / "composer_mapping.json"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "tokens_train.pt"
VAL_TOKENS_PATH = PROCESSED_DIR / "tokens_val.pt"

PITCH_RANGE = (21, 109)
NUM_VELOCITIES = 32
TOP_COMPOSERS = 15
OTHER_COMPOSER = "OTHER"
# Conditioning token dropped in during training so the model also learns the
# unconditional distribution required by classifier-free guidance at inference.
UNCONDITIONAL_COMPOSER = "UNCONDITIONAL"
RESERVED_COMPOSERS = (OTHER_COMPOSER, UNCONDITIONAL_COMPOSER)
# 12 positions per beat -> triplet / rubato-friendly 1/12 quantization.
POSITIONS_PER_BEAT = 12

# BPE target vocabulary. Merges frequent note/rhythm patterns into single ids,
# so a fixed 4096-token window covers substantially more music.
BPE_VOCAB_SIZE = 2048

# Pitch-shift augmentation range (semitones). BPE ids cannot be transposed
# directly (one id may span several notes), so every transposition is encoded
# at tokenization time and train.py picks a variant per document per sample.
MIN_PITCH_SHIFT = -6
MAX_PITCH_SHIFT = 5
TRAIN_SHIFTS = tuple(range(MIN_PITCH_SHIFT, MAX_PITCH_SHIFT + 1))
VAL_SHIFTS = (0,)

DOCS_FORMAT = "docs_v2_bpe"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MidiRecord:
    midi_path: Path
    composer: str
    split: str


def sanitize_composer_name(composer: str) -> str:
    normalized = unicodedata.normalize("NFKD", composer)
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", without_accents).strip("_")
    if not sanitized:
        raise ValueError(f"Composer name cannot be sanitized: {composer!r}")
    return sanitized


def composer_vocab_token(composer: str) -> str:
    return f"Composer_{sanitize_composer_name(composer)}"


class ComposerREMI(REMI):
    def _create_base_vocabulary(self) -> list[str]:
        vocabulary = super()._create_base_vocabulary()
        composer_tokens = self.config.additional_params.get("composer_tokens", [])
        return [*composer_tokens, *vocabulary]


def load_metadata(metadata_path: Path, dataset_dir: Path) -> pd.DataFrame:
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


def learned_token_id(tokenizer: REMI, token: str) -> int:
    """Id of a single base token in the vocabulary the model actually sees.

    Special and composer tokens never appear inside BPE merges (they are absent
    from the MIDI training corpus), so each keeps an atomic id in the learned
    vocabulary. Looked up through the byte mapping because miditok's
    ``encode_token_ids`` is only defined for full musical sequences.
    """
    if token not in tokenizer.vocab:
        raise KeyError(f"Token {token!r} missing from base vocabulary")
    base_id = int(tokenizer.vocab[token])
    if not tokenizer.is_trained:
        return base_id

    byte_form = tokenizer._ids_to_bytes([base_id], as_one_str=True)
    learned_id = tokenizer.vocab_model.get(byte_form)
    if learned_id is None:
        raise ValueError(f"Token {token!r} is not atomic in the learned vocabulary")
    return int(learned_id)


def encode_base_ids_batch(
    tokenizer: REMI,
    base_ids_batch: list[list[int]],
) -> list[list[int]]:
    sequences = [TokSequence(ids=list(ids)) for ids in base_ids_batch]
    tokenizer.encode_token_ids(sequences)
    return [list(sequence.ids) for sequence in sequences]


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


def build_tokenizer(composer_groups: list[str] | None = None) -> ComposerREMI:
    groups = composer_groups or []
    composer_tokens = [composer_vocab_token(composer) for composer in groups]
    if len(composer_tokens) != len(set(composer_tokens)):
        raise ValueError("Composer names collide after sanitization")

    config = TokenizerConfig(
        pitch_range=PITCH_RANGE,
        beat_res={(0, 4): POSITIONS_PER_BEAT, (4, 12): POSITIONS_PER_BEAT},
        num_velocities=NUM_VELOCITIES,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        composer_tokens=composer_tokens,
        use_tempos=True,
        use_velocities=True,
        use_chords=True,
        use_rests=True,
    )
    return ComposerREMI(config)


def validate_midi_file(midi_path: Path) -> bool:
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as exc:
        logger.warning("Skipping invalid MIDI file %s: %s", midi_path, exc)
        return False

    note_count = sum(len(instrument.notes) for instrument in midi.instruments)
    if note_count == 0:
        logger.warning("Skipping MIDI file without notes: %s", midi_path)
        return False

    return True


def as_token_sequences(tokens: TokSequence | list[TokSequence]) -> list[TokSequence]:
    return tokens if isinstance(tokens, list) else [tokens]


def collect_split_records(
    metadata: pd.DataFrame,
    split: str,
) -> tuple[list[MidiRecord], int]:
    records: list[MidiRecord] = []
    missing_files = 0
    split_rows = metadata[metadata["split"].astype(str).str.lower() == split]
    for row in split_rows.itertuples(index=False):
        midi_path = Path(row.midi_path)
        if not midi_path.exists():
            logger.warning("Skipping missing MIDI file: %s", midi_path)
            missing_files += 1
            continue
        if not validate_midi_file(midi_path):
            continue
        records.append(
            MidiRecord(
                midi_path=midi_path,
                composer=str(row.canonical_composer),
                split=str(row.split),
            )
        )
    return records, missing_files


def chunk_base_ids(ids: list[int], max_seq_len: int, stride: int) -> list[list[int]]:
    # One slot per chunk is reserved for the composer prefix added in train.py.
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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(METADATA_PATH, DATASET_DIR)
    composer_mapping = build_composer_mapping(metadata)
    composer_groups = build_composer_groups(composer_mapping)

    logger.info(
        "Loaded %d metadata rows | composers=%d | conditioned groups=%d",
        len(metadata),
        len(composer_mapping),
        len(composer_groups),
    )

    train_records, train_missing = collect_split_records(metadata, "train")
    val_records, val_missing = collect_split_records(metadata, "validation")
    logger.info(
        "Valid MIDI files: train=%d (missing %d) | validation=%d (missing %d)",
        len(train_records),
        train_missing,
        len(val_records),
        val_missing,
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
        files_paths=[record.midi_path for record in train_records],
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
