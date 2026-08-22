#!/usr/bin/env bash
# Download the raw MIDI corpora into data/raw_midi/.
#
# - MAESTRO v3.0.0 (MIDI only + metadata csv): automated.
# - GiantMIDI-Piano: manual download (distributed via Google Drive), see below.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data/raw_midi"
mkdir -p "$DATA_DIR"

# --- MAESTRO v3.0.0 -------------------------------------------------------
if [ ! -f "$DATA_DIR/maestro-v3.0.0.csv" ]; then
    echo "Downloading MAESTRO v3.0.0 MIDI (~60 MB) ..."
    ZIP="$(mktemp /tmp/maestro-v3.XXXXXX.zip)"
    curl -L --fail -o "$ZIP" \
        https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
    # The zip contains a maestro-v3.0.0/ folder; flatten it so that
    # data/raw_midi/maestro-v3.0.0.csv and data/raw_midi/<year>/*.midi exist,
    # matching the paths scripts/tokenize_midi.py expects.
    STAGE="$(mktemp -d /tmp/maestro-extract.XXXXXX)"
    unzip -q "$ZIP" -d "$STAGE"
    cp -R "$STAGE"/maestro-v3.0.0/. "$DATA_DIR"/
    rm -rf "$ZIP" "$STAGE"
    echo "MAESTRO ready at $DATA_DIR"
else
    echo "MAESTRO already present, skipping."
fi

# --- GiantMIDI-Piano (optional) -------------------------------------------
# The transcribed 'surname' dataset is shared on Google Drive; there is no
# stable direct URL, so fetch it manually:
#   1. Request/download GiantMIDI-Piano (midis_aligned_first_checked) from
#      https://github.com/bytedance/GiantMIDI-Piano
#   2. Place the .mid files under data/raw_midi/giantmidi/
# Filenames must keep the original 'Last, First, Title, youtubeId.mid' format;
# scripts/tokenize_midi.py parses the composer from the name and applies its
# own quality gates. Training works with MAESTRO alone if you skip this.
if [ -d "$DATA_DIR/giantmidi" ]; then
    echo "GiantMIDI directory present: $(find "$DATA_DIR/giantmidi" -name '*.mid' | wc -l | tr -d ' ') files."
else
    echo "GiantMIDI not found — see instructions above to add data/raw_midi/giantmidi/ (optional)."
fi
