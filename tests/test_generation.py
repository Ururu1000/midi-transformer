import math
from pathlib import Path

import pretty_midi
import pytest
import torch

from musiclm.data.tokenizer import (
    UNCONDITIONAL_COMPOSER,
    build_tokenizer,
    learned_token_id,
)
from musiclm.evaluation.metrics import note_window_features, stuck_fraction
from musiclm.inference.priming import drop_primer, load_primer
from musiclm.inference.rerank import ALL_COMPOSERS, WINDOW_FEATURES, score_notes
from musiclm.inference.sampler import (
    apply_repetition_penalty,
    generate_batch,
    min_p_filter,
)
from musiclm.model import MusicTransformer

COMPOSER = "Frédéric Chopin"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer([COMPOSER, UNCONDITIONAL_COMPOSER])


def tiny_model(vocab_size: int) -> MusicTransformer:
    torch.manual_seed(0)
    model = MusicTransformer(
        vocab_size=vocab_size, d_model=32, nhead=4, num_layers=2, d_ff=64,
        max_seq_len=256, dropout=0.0,
    )
    return model.eval()


def scale_notes(count: int, step: float = 0.1) -> list[pretty_midi.Note]:
    pitches = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
    return [
        pretty_midi.Note(velocity=80, pitch=pitches[i % len(pitches)] + 12 * (i % 3),
                         start=i * step, end=i * step + 0.3)
        for i in range(count)
    ]


def trill_notes(count: int, step: float = 0.1) -> list[pretty_midi.Note]:
    return [
        pretty_midi.Note(velocity=80, pitch=60 + 2 * (i % 2), start=i * step,
                         end=i * step + 0.1)
        for i in range(count)
    ]


def write_midi(path: Path, bars_of_notes: int, lead_in_bars: int = 0) -> None:
    # 120 BPM, 4/4: one bar is 2 seconds.
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    for bar in range(bars_of_notes):
        start = (lead_in_bars + bar) * 2.0
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60 + bar, start=start, end=start + 1.0))
    pm.instruments.append(piano)
    pm.write(str(path))


class TestBatchedSamplingOps:
    def test_min_p_rows_are_independent(self):
        rows = torch.tensor([[10.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
        filtered = min_p_filter(rows, min_p=0.5)
        assert torch.equal(filtered[0], min_p_filter(rows[0], 0.5))
        assert torch.isfinite(filtered[1]).all()

    def test_repetition_penalty_batch_matches_rows(self):
        logits = torch.tensor([[4.0, -4.0, 2.0, 1.0], [1.0, 3.0, -1.0, 2.0]])
        windows = torch.tensor([[0, 1], [3, 3]])
        is_pitch = torch.tensor([True, True, False, True])
        batched = apply_repetition_penalty(logits.clone(), windows, 2.0, is_pitch)
        for row in range(2):
            single = apply_repetition_penalty(logits[row].clone(), windows[row], 2.0, is_pitch)
            assert torch.equal(batched[row], single)


class TestGenerateBatch:
    def test_prefix_and_shapes(self, tokenizer):
        model = tiny_model(len(tokenizer))
        prefix = [tokenizer.vocab["Bar_None"], tokenizer.vocab["Position_0"]]
        samples = generate_batch(
            model, tokenizer, torch.device("cpu"), 40, composer=COMPOSER,
            num_samples=3, min_new_tokens=40, seed=0, prefix_ids=prefix,
        )
        assert len(samples) == 3
        composer_id = learned_token_id(tokenizer, "Composer_Frederic_Chopin")
        for sample in samples:
            assert sample[: 1 + len(prefix)] == [composer_id, *prefix]
            assert len(sample) == 1 + len(prefix) + 40

    def test_samples_differ_and_progress_reported(self, tokenizer):
        model = tiny_model(len(tokenizer))
        calls = []
        samples = generate_batch(
            model, tokenizer, torch.device("cpu"), 30, composer=COMPOSER,
            num_samples=2, min_new_tokens=30, seed=1, temperature=1.5, min_p=0.0,
            progress_callback=lambda done, total: calls.append((done, total)),
        )
        assert samples[0] != samples[1]
        assert calls[-1] == (30, 30)

    def test_prompt_too_long_raises(self, tokenizer):
        model = tiny_model(len(tokenizer))
        with pytest.raises(ValueError, match="no room"):
            generate_batch(
                model, tokenizer, torch.device("cpu"), 10, composer=COMPOSER,
                prefix_ids=[tokenizer.vocab["Bar_None"]] * 300,
            )


class TestPriming:
    def test_primer_trims_silence_and_cuts_by_bars(self, tokenizer, tmp_path):
        midi = tmp_path / "primer.mid"
        write_midi(midi, bars_of_notes=8, lead_in_bars=2)
        primer = load_primer(tokenizer, midi, bars=3)
        bar_id = tokenizer.vocab["Bar_None"]
        tokens = [tokenizer[i] for i in primer]
        assert primer[0] == bar_id and primer[-1] == bar_id
        # One note per bar: exactly the first three, with no leading rest.
        assert [t for t in tokens if t.startswith("Pitch_")] == ["Pitch_60", "Pitch_61", "Pitch_62"]
        first_event = next(t for t in tokens[1:] if not t.startswith(("Position_", "Tempo_")))
        assert first_event == "Pitch_60"

    def test_drop_primer_keeps_composer_and_bar(self, tokenizer):
        bar = learned_token_id(tokenizer, "Bar_None")
        ids = [7, 100, 101, 102, 55, 56]
        assert drop_primer(ids, primer_len=3, tokenizer=tokenizer) == [7, bar, 55, 56]

    def test_primer_without_notes_raises(self, tokenizer, tmp_path):
        midi = tmp_path / "empty.mid"
        pretty_midi.PrettyMIDI().write(str(midi))
        with pytest.raises(ValueError, match="no pitched notes"):
            load_primer(tokenizer, midi, bars=4)


class TestReranking:
    def test_stuck_fraction_flags_trill(self):
        assert stuck_fraction(trill_notes(200)) == 1.0
        assert stuck_fraction(scale_notes(200)) == 0.0

    def test_score_prefers_varied_music_without_stats(self):
        assert score_notes(scale_notes(200), COMPOSER, None) < score_notes(
            trill_notes(200), COMPOSER, None
        )

    def test_stuck_penalty_counts_only_excess_over_composer(self):
        features = note_window_features(trill_notes(200))
        reference = {"mean": features, "std": {n: 1.0 for n in WINDOW_FEATURES}}
        loose = {"window_notes": 600, "composers": {ALL_COMPOSERS: {**reference, "stuck_fraction": 1.0}}}
        strict = {"window_notes": 600, "composers": {ALL_COMPOSERS: {**reference, "stuck_fraction": 0.0}}}
        assert score_notes(trill_notes(200), COMPOSER, loose) == pytest.approx(0.0)
        assert score_notes(trill_notes(200), COMPOSER, strict) > 1.0

    def test_too_few_notes_is_worst(self):
        assert math.isinf(score_notes(scale_notes(10), COMPOSER, None))

    def test_score_uses_composer_stats_with_fallback(self):
        features = note_window_features(scale_notes(200))
        exact = {"mean": features, "std": {n: 1.0 for n in WINDOW_FEATURES}, "stuck_fraction": 0.0}
        far = {"mean": {n: v + 10.0 for n, v in features.items()},
               "std": {n: 1.0 for n in WINDOW_FEATURES}, "stuck_fraction": 0.0}
        stats = {"window_notes": 600, "composers": {COMPOSER: exact, ALL_COMPOSERS: far}}
        assert score_notes(scale_notes(200), COMPOSER, stats) == pytest.approx(0.0)
        assert score_notes(scale_notes(200), "Unknown", stats) == pytest.approx(10.0)
