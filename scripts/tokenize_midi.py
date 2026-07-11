from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pretty_midi
import torch
from miditok import REMI, TokSequence, TokenizerConfig
from torch import Tensor
from torch.utils.data import Dataset

MAX_SEQ_LEN = 2048
CHUNK_STRIDE = 1024
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
# 4 positions per beat -> a 4/4 bar is quantized into 16 (1/16) positions.
POSITIONS_PER_BEAT = 4

# Pitch-shift augmentation range (semitones), applied on the fly per sample.
MIN_PITCH_SHIFT = -6
MAX_PITCH_SHIFT = 5

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MidiRecord:
    midi_path: Path
    composer: str
    split: str


def composer_prompt(composer: str) -> str:
    return f"[COMPOSER: {composer}]"


def composer_vocab_token(composer: str) -> str:
    return f"{composer_prompt(composer)}_None"


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
    top_composers = set(counts.head(TOP_COMPOSERS).index)
    return {
        composer: composer if composer in top_composers else OTHER_COMPOSER
        for composer in counts.index
    }


def save_composer_mapping(mapping: dict[str, str], output_path: Path) -> None:
    payload = {
        "top_composers": [
            composer for composer, group in mapping.items() if group != OTHER_COMPOSER
        ],
        "other_token": composer_vocab_token(OTHER_COMPOSER),
        "composer_to_group": mapping,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_pitch_shift_maps(
    tokenizer: REMI,
    vocab_size: int,
    min_shift: int = MIN_PITCH_SHIFT,
    max_shift: int = MAX_PITCH_SHIFT,
) -> dict[int, Tensor]:
    pitch_token_ids = {
        int(token.split("_")[1]): token_id
        for token, token_id in tokenizer.vocab.items()
        if token.startswith("Pitch_")
    }
    assert pitch_token_ids, "No Pitch_* tokens found in tokenizer vocabulary"

    lowest, highest = min(pitch_token_ids), max(pitch_token_ids)
    shift_maps: dict[int, Tensor] = {}
    for shift in range(min_shift, max_shift + 1):
        # Identity mapping; only Pitch ids are redirected to the transposed pitch.
        mapping = torch.arange(vocab_size, dtype=torch.long)
        for pitch, token_id in pitch_token_ids.items():
            # Clamp so transposition never leaves the tokenizer's pitch range.
            shifted = min(max(pitch + shift, lowest), highest)
            mapping[token_id] = pitch_token_ids[shifted]
        shift_maps[shift] = mapping
    return shift_maps


class MusicDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pitch_shift_maps: dict[int, Tensor] | None = None,
        min_shift: int = MIN_PITCH_SHIFT,
        max_shift: int = MAX_PITCH_SHIFT,
    ) -> None:
        assert input_ids.ndim == 2, f"Got {input_ids.shape}"
        assert attention_mask.shape == input_ids.shape, (
            f"Got input_ids={input_ids.shape}, attention_mask={attention_mask.shape}"
        )

        self.input_ids = input_ids.long()
        self.attention_mask = attention_mask.bool()
        self.seq_len = input_ids.shape[1]
        self.pitch_shift_maps = pitch_shift_maps
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sequence = self.input_ids[index]
        mask = self.attention_mask[index]
        if self.pitch_shift_maps is not None:
            shift = random.randint(self.min_shift, self.max_shift)
            sequence = self.pitch_shift_maps[shift][sequence]

        assert sequence.shape == (self.seq_len,), f"Got {sequence.shape}"
        assert mask.shape == (self.seq_len,), f"Got {mask.shape}"
        return sequence, mask


def build_tokenizer(composer_groups: list[str] | None = None) -> REMI:
    groups = composer_groups or []
    config = TokenizerConfig(
        pitch_range=PITCH_RANGE,
        beat_res={(0, 4): POSITIONS_PER_BEAT, (4, 12): POSITIONS_PER_BEAT},
        num_velocities=NUM_VELOCITIES,
        special_tokens=[
            "PAD",
            "BOS",
            "EOS",
            "MASK",
            *(composer_prompt(composer) for composer in groups),
        ],
        use_tempos=True,
        use_velocities=True,
        use_chords=False,
        use_rests=False,
    )
    return REMI(config)


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


def sequence_ids(tokenizer: REMI, sequence: TokSequence) -> list[int]:
    tokenizer.complete_sequence(sequence)
    ids = sequence.ids
    if len(ids) == 0:
        return []

    assert all(isinstance(token_id, int) for token_id in ids), f"Got {type(ids[0])}"
    return ids


def chunk_ids(
    ids: list[int],
    max_seq_len: int,
    pad_token_id: int,
    stride: int,
    prefix_token_id: int,
) -> list[tuple[list[int], list[int]]]:
    content_length = max_seq_len - 1
    assert 0 < stride <= content_length, (
        f"stride={stride} must be in range [1, {content_length}]"
    )
    chunks: list[tuple[list[int], list[int]]] = []

    for start in range(0, len(ids), stride):
        content = ids[start : start + content_length]
        if len(content) == 0:
            continue

        chunk = [prefix_token_id, *content]
        attention_mask = [1] * len(chunk)
        pad_length = max_seq_len - len(chunk)
        if pad_length > 0:
            chunk = [*chunk, *([pad_token_id] * pad_length)]
            attention_mask = [*attention_mask, *([0] * pad_length)]

        assert len(chunk) == max_seq_len, f"Got {len(chunk)}"
        assert len(attention_mask) == max_seq_len, f"Got {len(attention_mask)}"
        chunks.append((chunk, attention_mask))

    return chunks


def tokenize_split(
    metadata: pd.DataFrame,
    split: str,
    tokenizer: REMI,
    composer_mapping: dict[str, str],
    max_seq_len: int,
    chunk_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    split_rows = metadata[metadata["split"].astype(str).str.lower() == split]
    processed_files = 0
    missing_files = 0

    for row in split_rows.itertuples(index=False):
        record = MidiRecord(
            midi_path=Path(row.midi_path),
            composer=str(row.canonical_composer),
            split=str(row.split),
        )
        midi_path = record.midi_path
        if not midi_path.exists():
            logger.warning("Skipping missing MIDI file: %s", midi_path)
            missing_files += 1
            continue
        if not validate_midi_file(midi_path):
            continue

        composer_group = composer_mapping.get(record.composer, OTHER_COMPOSER)
        prefix_token_id = tokenizer[composer_vocab_token(composer_group)]
        try:
            tokens = tokenizer(midi_path)
        except Exception as exc:
            logger.warning("Skipping MIDI file %s: tokenization failed: %s", midi_path, exc)
            continue

        for sequence in as_token_sequences(tokens):
            ids = sequence_ids(tokenizer, sequence)
            for chunk, attention_mask in chunk_ids(
                ids,
                max_seq_len,
                tokenizer.pad_token_id,
                chunk_stride,
                prefix_token_id,
            ):
                all_input_ids.append(chunk)
                all_attention_masks.append(attention_mask)
        processed_files += 1

    if len(all_input_ids) == 0:
        msg = f"No token sequences were created for split '{split}'."
        raise ValueError(msg)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_masks, dtype=torch.bool)

    expected_shape = (len(all_input_ids), max_seq_len)
    assert input_ids.shape == expected_shape, f"Got {input_ids.shape}"
    assert attention_mask.shape == expected_shape, f"Got {attention_mask.shape}"

    logger.info(
        "Processed split=%s | files=%d | missing=%d | chunks=%d",
        split,
        processed_files,
        missing_files,
        len(all_input_ids),
    )
    return input_ids, attention_mask


def save_tokenized_tensors(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: REMI,
    output_path: Path,
) -> None:
    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pad_token_id": tokenizer.pad_token_id,
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
    composer_groups = [
        composer
        for composer, group in composer_mapping.items()
        if group != OTHER_COMPOSER
    ]
    composer_groups.append(OTHER_COMPOSER)

    logger.info(
        "Loaded %d metadata rows | composers=%d | conditioned groups=%d",
        len(metadata),
        len(composer_mapping),
        len(composer_groups),
    )

    tokenizer = build_tokenizer(composer_groups)
    tokenizer.save(TOKENIZER_PATH)
    save_composer_mapping(composer_mapping, COMPOSER_MAPPING_PATH)

    train_input_ids, train_attention_mask = tokenize_split(
        metadata,
        "train",
        tokenizer,
        composer_mapping,
        MAX_SEQ_LEN,
        CHUNK_STRIDE,
    )
    val_input_ids, val_attention_mask = tokenize_split(
        metadata,
        "validation",
        tokenizer,
        composer_mapping,
        MAX_SEQ_LEN,
        CHUNK_STRIDE,
    )

    save_tokenized_tensors(
        train_input_ids,
        train_attention_mask,
        tokenizer,
        TRAIN_TOKENS_PATH,
    )
    save_tokenized_tensors(
        val_input_ids,
        val_attention_mask,
        tokenizer,
        VAL_TOKENS_PATH,
    )
    logger.info(
        "Saved train=%d to %s | validation=%d to %s | tokenizer=%s",
        len(train_input_ids),
        TRAIN_TOKENS_PATH,
        len(val_input_ids),
        VAL_TOKENS_PATH,
        TOKENIZER_PATH,
    )


if __name__ == "__main__":
    main()
