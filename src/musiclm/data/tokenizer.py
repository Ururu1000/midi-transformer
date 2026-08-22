"""Composer-conditioned REMI tokenizer and vocabulary helpers.

This module is the only place allowed to touch miditok internals (private
attributes such as ``_vocab_learned_bytes_to_tokens``); everything else in the
package goes through the functions defined here.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from miditok import REMI, TokenizerConfig, TokSequence

# 12 positions per beat -> triplet / rubato-friendly 1/12 quantization.
POSITIONS_PER_BEAT = 12
PITCH_RANGE = (21, 109)
NUM_VELOCITIES = 32

TOP_COMPOSERS = 15
OTHER_COMPOSER = "OTHER"
# Conditioning token dropped in during training so the model also learns the
# unconditional distribution required by classifier-free guidance at inference.
UNCONDITIONAL_COMPOSER = "UNCONDITIONAL"
RESERVED_COMPOSERS = (OTHER_COMPOSER, UNCONDITIONAL_COMPOSER)

logger = logging.getLogger(__name__)


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


def iter_vocab_entries(tokenizer: REMI) -> list[tuple[int, list[str]]]:
    """(model id, constituent base tokens) for every id the model can emit.

    With a trained BPE model one id may decompose into several base tokens;
    without one every id maps to exactly one base token.
    """
    if tokenizer.is_trained:
        bytes_to_tokens = tokenizer._vocab_learned_bytes_to_tokens
        return [
            (int(token_id), list(bytes_to_tokens[byte_form]))
            for byte_form, token_id in tokenizer.vocab_model.items()
        ]
    return [(int(token_id), [token]) for token, token_id in tokenizer.vocab.items()]


def list_composer_tokens(tokenizer: REMI) -> list[str]:
    return sorted(token for token in tokenizer.vocab if token.startswith("Composer_"))
