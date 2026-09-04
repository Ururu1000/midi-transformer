#!/usr/bin/env bash
# One-time setup for MIDI -> MP3 conversion:
#   1. FluidSynth binary (Homebrew) for real General-MIDI piano rendering
#   2. FreePats YDP Grand Piano SoundFont into data/soundfonts/
#
# Idempotent: re-running skips whatever is already installed/downloaded.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- 1. FluidSynth binary --------------------------------------------------
if command -v fluidsynth >/dev/null 2>&1; then
    echo "fluidsynth already installed: $(command -v fluidsynth)"
elif command -v brew >/dev/null 2>&1; then
    echo "Installing fluidsynth via Homebrew ..."
    brew install fluidsynth
else
    echo "ERROR: fluidsynth is missing and Homebrew was not found." >&2
    echo "Install it manually, e.g. 'apt install fluidsynth' on Linux." >&2
    exit 1
fi

# --- 2. Studio SoundFont ---------------------------------------------------
if [ -n "${PYTHON:-}" ]; then
    PYTHON_BIN="$PYTHON"
elif [ -x "$ROOT/venv/bin/python" ]; then
    PYTHON_BIN="$ROOT/venv/bin/python"
elif [ -x "$ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi
echo "Installing verified FreePats YDP Grand Piano SoundFont ..."
SF_PATH="$(
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c \
        'from musiclm.audio import download_studio_soundfont; print(download_studio_soundfont())'
)"

echo
echo "Done. MP3 rendering will use:"
echo "  fluidsynth : $(command -v fluidsynth)"
echo "  soundfont  : $SF_PATH"
