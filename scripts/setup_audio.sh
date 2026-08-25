#!/usr/bin/env bash
# One-time setup for MIDI -> MP3 conversion:
#   1. FluidSynth binary (Homebrew) for real General-MIDI piano rendering
#   2. FluidR3_GM SoundFont into data/soundfonts/
#
# Idempotent: re-running skips whatever is already installed/downloaded.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SF_DIR="$ROOT/data/soundfonts"
SF_NAME="FluidR3_GM.sf2"

mkdir -p "$SF_DIR"

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

# --- 2. SoundFont ----------------------------------------------------------
SF_PATH="$SF_DIR/$SF_NAME"
if [ -s "$SF_PATH" ]; then
    echo "SoundFont already present: $SF_PATH ($(du -h "$SF_PATH" | cut -f1))"
else
    echo "Fetching FluidR3_GM SoundFont (~140 MB) ..."
    tmp="$(mktemp /tmp/FluidR3_GM.XXXXXX.sf2)"
    ok=0

    # Source A: direct download (works when not behind Cloudflare bot checks).
    if curl -L --fail --retry 2 --progress-bar \
        -o "$tmp" "https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2" \
        && head -c 4 "$tmp" | grep -q RIFF; then
        ok=1
    fi

    # Source B: Debian ships the same FluidR3_GM.sf2 inside a .deb package.
    if [ "$ok" != "1" ]; then
        echo "  falling back to the Debian package mirror ..." >&2
        work="$(mktemp -d /tmp/fluidsf.XXXXXX)"
        deb_url="http://deb.debian.org/debian/pool/main/f/fluid-soundfont"
        # Pick the newest revision listed in the pool directory.
        deb_name="$(curl -s --fail "$deb_url/" | grep -oE 'fluid-soundfont-gm_[^"]+\.deb' | sort -V | tail -1)"
        curl -L --fail --retry 2 --progress-bar -o "$work/pkg.deb" "$deb_url/$deb_name"
        (cd "$work" && ar x pkg.deb data.tar.xz && tar -xf data.tar.xz)
        candidate="$(find "$work" -name '*.sf2' | head -1)"
        if [ -n "$candidate" ] && head -c 4 "$candidate" | grep -q RIFF; then
            cp "$candidate" "$tmp"
            ok=1
        fi
        rm -rf "$work"
    fi

    if [ "$ok" != "1" ]; then
        echo "All sources failed. Download FluidR3_GM.sf2 manually into $SF_DIR." >&2
        rm -f "$tmp"
        exit 1
    fi
    mv "$tmp" "$SF_PATH"
    echo "SoundFont saved to $SF_PATH"
fi

echo
echo "Done. MP3 rendering will use:"
echo "  fluidsynth : $(command -v fluidsynth)"
echo "  soundfont  : $SF_PATH"
