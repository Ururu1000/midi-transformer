"""MIDI rendering and MP3 encoding.

Rendering tiers (first available wins):
  1. FluidSynth CLI + SoundFont  – real General-MIDI piano
  2. pretty_midi.fluidsynth()    – pyfluidsynth package + SoundFont
  3. pretty_midi.synthesize()    – sine waves; always works

MP3 encoding:
  lameenc (bundled-LAME pip wheel) or ffmpeg when installed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pretty_midi

from musiclm.config import SOUNDFONT_DIR

SAMPLE_RATE = 44100
DEFAULT_BITRATE = "192k"
STUDIO_SOUNDFONT_NAME = "YDP-GrandPiano-20160804.sf2"
STUDIO_SOUNDFONT_URL = (
    "https://freepats.zenvoid.org/Piano/YDP-GrandPiano/"
    "YDP-GrandPiano-SF2-20160804.tar.bz2"
)
STUDIO_SOUNDFONT_SHA256 = "d243dc3e182a60df2a16e92828c1821cf3eb5748b45e2e2bdcfa9cf7af056026"
STUDIO_SOUNDFONT_LICENSE = "CC BY 3.0"
_STUDIO_SOUNDFONT_MEMBER = (
    "YDP-GrandPiano-SF2-20160804/YDP-GrandPiano-20160804.sf2"
)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_AUTO_DOWNLOAD_FALSE_VALUES = {"0", "false", "no", "off"}
_soundfont_download_lock = threading.Lock()
_auto_download_failed = False

# Known system locations for soundfonts (brew on macOS, apt on Linux).
_SYSTEM_SF2_DIRS = (
    Path("/opt/homebrew/share/soundfonts"),
    Path("/usr/local/share/soundfonts"),
    Path("/usr/share/sounds/sf2"),
)

logger = logging.getLogger(__name__)


class SoundFontDownloadError(RuntimeError):
    """Raised when the managed studio SoundFont cannot be installed safely."""


def _is_valid_soundfont(path: Path) -> bool:
    try:
        with path.open("rb") as soundfont:
            header = soundfont.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[:4] == b"RIFF" and header[8:12] == b"sfbk"


def _auto_download_enabled() -> bool:
    value = os.environ.get("MUSICLM_AUTO_DOWNLOAD_SOUNDFONT", "1")
    return value.strip().lower() not in _AUTO_DOWNLOAD_FALSE_VALUES


def download_studio_soundfont(destination_dir: Path | None = None) -> Path:
    """Download, verify, and atomically install the FreePats YDP piano SF2.

    YDP Grand Piano is built from Zenph Studios Yamaha Disklavier Pro samples
    and published by FreePats under CC BY 3.0. The compressed archive is about
    35 MiB; the extracted SoundFont is cached for all future renders.
    """
    destination_dir = Path(destination_dir or SOUNDFONT_DIR)
    target = destination_dir / STUDIO_SOUNDFONT_NAME
    if _is_valid_soundfont(target):
        return target

    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path: Path | None = None
    extracted_path: Path | None = None

    with _soundfont_download_lock:
        if _is_valid_soundfont(target):
            return target

        try:
            with tempfile.NamedTemporaryFile(
                prefix=".ydp-grand-piano-",
                suffix=".tar.bz2",
                dir=destination_dir,
                delete=False,
            ) as archive_file:
                archive_path = Path(archive_file.name)
                request = urllib.request.Request(
                    STUDIO_SOUNDFONT_URL,
                    headers={"User-Agent": "musiclm/0.1 SoundFont downloader"},
                )
                logger.info(
                    "Downloading studio piano SoundFont from FreePats "
                    "(35 MiB archive, first render only) ..."
                )
                digest = hashlib.sha256()
                with urllib.request.urlopen(request, timeout=60) as response:
                    while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                        archive_file.write(chunk)
                        digest.update(chunk)

            actual_digest = digest.hexdigest()
            if actual_digest != STUDIO_SOUNDFONT_SHA256:
                raise SoundFontDownloadError(
                    "Studio SoundFont checksum mismatch: "
                    f"expected {STUDIO_SOUNDFONT_SHA256}, got {actual_digest}"
                )

            with tarfile.open(archive_path, mode="r:bz2") as archive:
                try:
                    member = archive.getmember(_STUDIO_SOUNDFONT_MEMBER)
                except KeyError as exc:
                    raise SoundFontDownloadError(
                        "Studio SoundFont archive has no expected SF2 file"
                    ) from exc
                source = archive.extractfile(member)
                if source is None:
                    raise SoundFontDownloadError(
                        "Studio SoundFont archive entry is not a regular file"
                    )
                with source, tempfile.NamedTemporaryFile(
                    prefix=f".{STUDIO_SOUNDFONT_NAME}.",
                    suffix=".part",
                    dir=destination_dir,
                    delete=False,
                ) as extracted_file:
                    extracted_path = Path(extracted_file.name)
                    shutil.copyfileobj(source, extracted_file, _DOWNLOAD_CHUNK_SIZE)

            if not _is_valid_soundfont(extracted_path):
                raise SoundFontDownloadError(
                    "Downloaded studio SoundFont has an invalid SF2 header"
                )
            os.replace(extracted_path, target)
            extracted_path = None
            logger.info(
                "Installed %s (FreePats, %s) at %s",
                STUDIO_SOUNDFONT_NAME,
                STUDIO_SOUNDFONT_LICENSE,
                target,
            )
            return target
        except SoundFontDownloadError:
            raise
        except (OSError, tarfile.TarError, urllib.error.URLError) as exc:
            raise SoundFontDownloadError(
                f"Could not download studio SoundFont: {exc}"
            ) from exc
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            if extracted_path is not None:
                extracted_path.unlink(missing_ok=True)


def find_soundfont(
    explicit: str | os.PathLike[str] | None = None,
    *,
    auto_download: bool = False,
) -> Path | None:
    """Locate a .sf2 file; explicit argument wins over env and system paths.

    Search order: explicit argument -> $MUSICLM_SOUNDFONT ->
    managed studio piano -> data/soundfonts/*.sf2 -> known system directories.
    When *auto_download* is true, install the managed piano if needed.
    """
    global _auto_download_failed

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

    studio_path = SOUNDFONT_DIR / STUDIO_SOUNDFONT_NAME
    if _is_valid_soundfont(studio_path):
        return studio_path

    if auto_download and _auto_download_enabled() and not _auto_download_failed:
        try:
            return download_studio_soundfont()
        except SoundFontDownloadError as exc:
            _auto_download_failed = True
            logger.warning(
                "%s; using an installed SoundFont instead. Retry manually with "
                "`download_studio_soundfont()` or disable attempts with "
                "MUSICLM_AUTO_DOWNLOAD_SOUNDFONT=0.",
                exc,
            )

    if SOUNDFONT_DIR.exists():
        candidates = sorted(
            path
            for path in SOUNDFONT_DIR.glob("*.sf2")
            if _is_valid_soundfont(path)
        )
        if candidates:
            return candidates[0]

    for directory in _SYSTEM_SF2_DIRS:
        if directory.exists():
            candidates = sorted(
                path for path in directory.rglob("*.sf2") if _is_valid_soundfont(path)
            )
            if candidates:
                return candidates[0]

    return None


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Scale to [-1, 1] with headroom so playback does not clip."""
    if audio.size == 0:
        return audio
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
    fluidsynth_bin = shutil.which("fluidsynth")
    pyfluidsynth_available = importlib.util.find_spec("fluidsynth") is not None
    sf = find_soundfont(
        soundfont,
        auto_download=fluidsynth_bin is not None or pyfluidsynth_available,
    )
    wav_path.parent.mkdir(parents=True, exist_ok=True)

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
