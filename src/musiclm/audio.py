"""MIDI rendering and MP3 encoding.

Rendering tiers (first available wins):
  1. FluidSynth CLI + SoundFont  – real General-MIDI piano
  2. pretty_midi.fluidsynth()    – pyfluidsynth package + SoundFont
  3. pretty_midi.synthesize()    – sine waves; always works

MP3 encoding:
  lameenc (bundled-LAME pip wheel) or ffmpeg when installed.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pretty_midi

from musiclm.config import SOUNDFONT_DIR

SAMPLE_RATE = 44100
DEFAULT_BITRATE = "192k"

# Known system locations for soundfonts (brew on macOS, apt on Linux).
_SYSTEM_SF2_DIRS = (
    Path("/opt/homebrew/share/soundfonts"),
    Path("/usr/local/share/soundfonts"),
    Path("/usr/share/sounds/sf2"),
)

logger = logging.getLogger(__name__)


def find_soundfont(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate a .sf2 file; explicit argument wins over env and system paths.

    Search order: explicit argument -> $MUSICLM_SOUNDFONT ->
    data/soundfonts/*.sf2 -> known system directories.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"SoundFont not found: {path}")
        return path

    env_path = os.environ.get("MUSICLM_SOUNDFONT")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        logger.warning("$MUSICLM_SOUNDFONT=%s does not exist, ignoring", env_path)

    if SOUNDFONT_DIR.exists():
        candidates = sorted(SOUNDFONT_DIR.glob("*.sf2"))
        if candidates:
            return candidates[0]

    for directory in _SYSTEM_SF2_DIRS:
        if directory.exists():
            candidates = sorted(directory.rglob("*.sf2"))
            if candidates:
                return candidates[0]

    return None


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Scale to [-1, 1] with headroom so playback does not clip."""
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def render_midi_to_wav(
    midi_path: Path,
    wav_path: Path,
    sample_rate: int = SAMPLE_RATE,
    soundfont: str | os.PathLike[str] | None = None,
) -> str:
    """Render *midi_path* to a mono 16-bit WAV file; returns the engine used."""
    sf = find_soundfont(soundfont)
    fluidsynth_bin = shutil.which("fluidsynth")

    if fluidsynth_bin is not None and sf is not None:
        # -n no shell, -i no stdin prompt, -g gain below clipping threshold.
        subprocess.run(
            [
                fluidsynth_bin,
                "-ni",
                "-g", "0.7",
                "-r", str(sample_rate),
                "-F", str(wav_path),
                str(sf),
                str(midi_path),
            ],
            check=True,
            capture_output=True,
        )
        logger.info("Rendered %s via FluidSynth CLI (%s)", midi_path.name, sf.name)
        return "fluidsynth"

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    engine = "sine"
    try:
        if sf is not None:
            audio = pm.fluidsynth(fs=sample_rate, sf2_path=str(sf))
            engine = "pyfluidsynth"
        else:
            raise RuntimeError("no soundfont found")
    except Exception as exc:
        logger.warning(
            "FluidSynth rendering unavailable (%s); falling back to sine "
            "synthesis. For real piano sound run scripts/setup_audio.sh.",
            exc,
        )
        audio = pm.synthesize(fs=sample_rate)

    audio = _normalize(np.asarray(audio)).astype(np.float32)
    pcm = (audio * 32767.0).astype("<i2").tobytes()

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    logger.info("Rendered %s via %s -> %s", midi_path.name, engine, wav_path.name)
    return engine


def wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    bitrate: str = DEFAULT_BITRATE,
    sample_rate: int = SAMPLE_RATE,
) -> str:
    """Encode a WAV file to MP3; returns the encoder used ("ffmpeg"/"lameenc")."""
    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is not None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame",
                "-b:a", bitrate,
                str(mp3_path),
            ],
            check=True,
        )
        return "ffmpeg"

    try:
        import lameenc
    except ImportError as exc:
        raise ImportError(
            "MP3 encoding needs either ffmpeg on PATH or the lameenc package: "
            "pip install -e '.[audio]'"
        ) from exc

    kbps = int(bitrate.rstrip("kK"))
    encoder = lameenc.Encoder()

    # Read the actual format from the file: the FluidSynth CLI renders stereo,
    # the pretty_midi fallback writes mono.
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        actual_rate = wf.getframerate()
        pcm_frames = wf.readframes(wf.getnframes())

    encoder.set_bit_rate(kbps)
    encoder.set_in_sample_rate(actual_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)  # 2 = high

    mp3_bytes = encoder.encode(pcm_frames) + encoder.flush()
    mp3_path.write_bytes(bytes(mp3_bytes))
    return "lameenc"


def convert_midi_to_mp3(
    midi_path: Path,
    output: Path | None = None,
    bitrate: str = DEFAULT_BITRATE,
    sample_rate: int = SAMPLE_RATE,
    soundfont: str | os.PathLike[str] | None = None,
) -> Path:
    """Convert a MIDI file to MP3 next to it; the MIDI file itself is kept."""
    midi_path = Path(midi_path)
    if output is None:
        output = midi_path.with_suffix(".mp3")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        render_engine = render_midi_to_wav(
            midi_path, wav_path, sample_rate=sample_rate, soundfont=soundfont
        )
        encoder = wav_to_mp3(wav_path, output, bitrate=bitrate, sample_rate=sample_rate)
    finally:
        wav_path.unlink(missing_ok=True)

    size_kb = output.stat().st_size / 1024
    logger.info(
        "Converted %s -> %s (%.0f KB, %s @ %s)",
        midi_path.name,
        output.name,
        size_kb,
        f"render={render_engine}",
        f"encode={encoder}",
    )
    return output
