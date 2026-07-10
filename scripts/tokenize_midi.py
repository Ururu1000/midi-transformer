from __future__ import annotations

import logging
import random
from pathlib import Path

import pretty_midi
import torch
from miditok import REMI, TokSequence, TokenizerConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

MAX_SEQ_LEN = 2048
RAW_MIDI_DIR = Path("data/raw_midi")
PROCESSED_DIR = Path("data/processed")
TOKENIZER_PATH = PROCESSED_DIR / "tokenizer.json"
TOKENS_PATH = PROCESSED_DIR / "tokens.pt"

PITCH_RANGE = (21, 109)
NUM_VELOCITIES = 32
# 4 positions per beat -> a 4/4 bar is quantized into 16 (1/16) positions.
POSITIONS_PER_BEAT = 4

# Pitch-shift augmentation range (semitones), applied on the fly per sample.
MIN_PITCH_SHIFT = -6
MAX_PITCH_SHIFT = 5

logger = logging.getLogger(__name__)


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
        self.attention_mask = attention_mask.long()
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


def find_midi_files(raw_midi_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_midi_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )


def build_tokenizer() -> REMI:
    config = TokenizerConfig(
        pitch_range=PITCH_RANGE,
        beat_res={(0, 4): POSITIONS_PER_BEAT, (4, 12): POSITIONS_PER_BEAT},
        num_velocities=NUM_VELOCITIES,
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
) -> list[tuple[list[int], list[int]]]:
    chunks: list[tuple[list[int], list[int]]] = []

    for start in range(0, len(ids), max_seq_len):
        chunk = ids[start : start + max_seq_len]
        if len(chunk) == 0:
            continue

        attention_mask = [1] * len(chunk)
        pad_length = max_seq_len - len(chunk)
        if pad_length > 0:
            chunk = [*chunk, *([pad_token_id] * pad_length)]
            attention_mask = [*attention_mask, *([0] * pad_length)]

        assert len(chunk) == max_seq_len, f"Got {len(chunk)}"
        assert len(attention_mask) == max_seq_len, f"Got {len(attention_mask)}"
        chunks.append((chunk, attention_mask))

    return chunks


def tokenize_midi_files(
    midi_files: list[Path],
    tokenizer: REMI,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []

    for midi_path in midi_files:
        if not validate_midi_file(midi_path):
            continue

        tokens = tokenizer(midi_path)
        for sequence in as_token_sequences(tokens):
            ids = sequence_ids(tokenizer, sequence)
            for chunk, attention_mask in chunk_ids(
                ids,
                max_seq_len,
                tokenizer.pad_token_id,
            ):
                all_input_ids.append(chunk)
                all_attention_masks.append(attention_mask)

    if len(all_input_ids) == 0:
        msg = f"No token sequences were created from {len(midi_files)} MIDI files."
        raise ValueError(msg)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(all_attention_masks, dtype=torch.long)

    expected_shape = (len(all_input_ids), max_seq_len)
    assert input_ids.shape == expected_shape, f"Got {input_ids.shape}"
    assert attention_mask.shape == expected_shape, f"Got {attention_mask.shape}"

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
            "max_seq_len": MAX_SEQ_LEN,
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    midi_files = find_midi_files(RAW_MIDI_DIR)
    if len(midi_files) == 0:
        msg = f"No .mid or .midi files found in {RAW_MIDI_DIR}."
        raise FileNotFoundError(msg)

    logger.info("Found %d MIDI files in %s", len(midi_files), RAW_MIDI_DIR)

    tokenizer = build_tokenizer()
    tokenizer.save(TOKENIZER_PATH)

    input_ids, attention_mask = tokenize_midi_files(
        midi_files,
        tokenizer,
        MAX_SEQ_LEN,
    )

    pitch_shift_maps = build_pitch_shift_maps(tokenizer, len(tokenizer))
    dataset = MusicDataset(input_ids, attention_mask, pitch_shift_maps)
    batch_size = min(4, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    first_ids, first_mask = next(iter(dataloader))
    first_ids = first_ids.to(device)
    first_mask = first_mask.to(device)

    expected_batch_shape = (batch_size, MAX_SEQ_LEN)
    assert first_ids.shape == expected_batch_shape, f"Got {first_ids.shape}"
    assert first_mask.shape == expected_batch_shape, f"Got {first_mask.shape}"

    save_tokenized_tensors(input_ids, attention_mask, tokenizer, TOKENS_PATH)
    logger.info(
        "Saved %d sequences to %s and tokenizer params to %s",
        len(dataset),
        TOKENS_PATH,
        TOKENIZER_PATH,
    )


if __name__ == "__main__":
    main()
