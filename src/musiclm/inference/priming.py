"""Primers: seed generation with the opening bars of a real MIDI file.

A primer pins key, meter and texture before the model samples anything. It
is cut by time into whole bars and always ends on a bar line, so the
continuation starts cleanly and the primer can be removed afterwards.

Starting from the piece-start tokens (``Bar``, ``Position_0``, ``Tempo``)
instead of a bare composer token was measured and rejected: the model then
writes sparse, opening-like material (some pieces came out at a third of
real Chopin's note density).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pretty_midi
from miditok import REMI

from musiclm.data.tokenizer import encode_base_ids_batch, learned_token_id

BAR_TOKEN = "Bar_None"
# MAESTRO performance MIDI is written at 120 BPM in 4/4.
DEFAULT_TEMPO_BPM = 120.0
BEATS_PER_BAR = 4


def _primer_midi(midi_path: Path, out_path: Path, bars: int) -> None:
    """Write the first *bars* bars after the first note as one piano track.

    Bars are cut by time, not by ``Bar`` tokens: with rests enabled REMI skips
    empty bar lines, so token counts do not match musical bars.
    """
    source = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [
        note
        for instrument in source.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    if not notes:
        raise ValueError(f"Primer MIDI has no pitched notes: {midi_path}")

    first_onset = min(note.start for note in notes)
    cutoff = bars * BEATS_PER_BAR * 60.0 / DEFAULT_TEMPO_BPM
    piano = pretty_midi.Instrument(program=0)
    for note in notes:
        start = note.start - first_onset
        if start < cutoff:
            piano.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=start,
                end=min(note.end - first_onset, cutoff),
            ))
    primer = pretty_midi.PrettyMIDI(initial_tempo=DEFAULT_TEMPO_BPM)
    primer.instruments.append(piano)
    primer.write(str(out_path))


def load_primer(tokenizer: REMI, midi_path: str | Path, bars: int) -> list[int]:
    """Model ids for the first *bars* 4/4 bars of *midi_path*, from its first note.

    Leading silence is trimmed. The primer always ends with a ``Bar`` token,
    so generation continues from the top of the next bar and the primer can
    later be cut off cleanly with :func:`drop_primer`.
    """
    assert bars > 0, f"Got bars={bars}"
    with tempfile.TemporaryDirectory() as tmp:
        primer_path = Path(tmp) / "primer.mid"
        _primer_midi(Path(midi_path), primer_path, bars)
        encoded = tokenizer.encode(primer_path, encode_ids=False)
    sequence = encoded[0] if isinstance(encoded, list) else encoded
    if not any(token.startswith("Pitch_") for token in sequence.tokens):
        raise ValueError(f"Primer MIDI has no notes in tokenizer range: {midi_path}")

    bar_id = tokenizer.vocab[BAR_TOKEN]
    return encode_base_ids_batch(tokenizer, [[*sequence.ids, bar_id]])[0]


def drop_primer(token_ids: list[int], primer_len: int, tokenizer: REMI) -> list[int]:
    """Keep the composer token, replace the primer with a single bar line."""
    bar_id = learned_token_id(tokenizer, BAR_TOKEN)
    return [token_ids[0], bar_id, *token_ids[1 + primer_len :]]
