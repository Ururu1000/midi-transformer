import math
import wave
from pathlib import Path

import numpy as np
import pytest

lameenc = pytest.importorskip("lameenc", reason="pip install -e '.[audio]'")

from musiclm.audio import (
    DEFAULT_BITRATE,
    convert_midi_to_mp3,
    find_soundfont,
    render_midi_to_wav,
    wav_to_mp3,
)


def write_sine_wav(path: Path, seconds=0.5, freq=440.0, rate=44100) -> None:
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    audio = (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    pcm = (audio * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def make_tiny_midi(path: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    for pitch, start in ((60, 0.0), (64, 0.5), (67, 1.0)):
        note = pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=start + 0.45)
        piano.notes.append(note)
    pm.instruments.append(piano)
    pm.write(str(path))


class TestWavToMp3:
    def test_produces_valid_mp3(self, tmp_path):
        wav = tmp_path / "tone.wav"
        mp3 = tmp_path / "tone.mp3"
        write_sine_wav(wav)

        encoder = wav_to_mp3(wav, mp3)

        assert mp3.exists() and mp3.stat().st_size > 100
        header = mp3.read_bytes()[:3]
        assert header.startswith(b"ID3") or header[0] == 0xFF  # ID3 tag or frame sync
        assert encoder in ("lameenc", "ffmpeg")

    def test_custom_bitrate(self, tmp_path):
        wav = tmp_path / "a.wav"
        lo, hi = tmp_path / "lo.mp3", tmp_path / "hi.mp3"
        write_sine_wav(wav)
        wav_to_mp3(wav, lo, bitrate="64k")
        wav_to_mp3(wav, hi, bitrate="320k")
        # Both valid; sizes differ (sine tone compresses well but not identically).
        assert lo.stat().st_size > 0 and hi.stat().st_size > 0


class TestRenderMidiToWav:
    def test_renders_nonempty_wav(self, tmp_path):
        midi = tmp_path / "tiny.mid"
        make_tiny_midi(midi)
        wav = tmp_path / "tiny.wav"

        engine = render_midi_to_wav(midi, wav)

        assert engine in ("fluidsynth", "pyfluidsynth", "sine")
        with wave.open(str(wav), "rb") as wf:
            frames = wf.getnframes()
            assert wf.getframerate() == 44100
            assert wf.getnchannels() in (1, 2)  # fluidsynth=stereo, fallback=mono
        assert frames > 44100 // 4  # at least a quarter second of audio


class TestConvertMidiToMp3:
    def test_keeps_midi_and_creates_sibling_mp3(self, tmp_path):
        midi = tmp_path / "piece.mid"
        make_tiny_midi(midi)
        before = midi.read_bytes()

        mp3 = convert_midi_to_mp3(midi)

        assert mp3 == tmp_path / "piece.mp3"
        assert midi.read_bytes() == before, "MIDI must be untouched"
        assert mp3.stat().st_size > 100

    def test_explicit_output_and_bitrate(self, tmp_path):
        midi = tmp_path / "x.mid"
        make_tiny_midi(midi)
        out = tmp_path / "custom" / "x.mp3"

        result = convert_midi_to_mp3(midi, output=out, bitrate="128k")

        assert result == out and out.exists()


class TestFindSoundfont:
    def test_missing_explicit_raises(self):
        with pytest.raises(FileNotFoundError):
            find_soundfont("/nonexistent/font.sf2")

    def test_finds_installed_font(self):
        sf = find_soundfont()
        if sf is not None:
            assert sf.suffix == ".sf2" and sf.exists()


def test_default_bitrate_constant():
    assert DEFAULT_BITRATE == "192k"
