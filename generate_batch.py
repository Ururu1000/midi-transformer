"""Generate 5 MIDI files with varied sampling parameters for A/B listening."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from miditok import REMI

from generate import generate, load_model, tokens_to_midi_file
from model import get_device
from scripts.tokenize_midi import ComposerREMI

CHECKPOINT_PATH = Path("checkpoints/model_epoch_20.pt")
TOKENIZER_PATH = Path("data/processed/tokenizer.json")
OUTPUT_DIR = Path("data/generated")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Preset:
    name: str
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    length: int
    description: str


PRESETS: list[Preset] = [
    Preset(
        name="01",
        temperature=1.08,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.24,
        length=512,
        description="Low temp, tight nucleus, strong anti-repeat → clean melody",
    ),
    Preset(
        name="02",
        temperature=1.08,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.24,
        length=512,
        description="Balanced sampling with nucleus filtering, natural phrasing",
    ),
    Preset(
        name="03",
        temperature=1.08,
        top_k=60,   
        top_p=0.95,
        repetition_penalty=1.24,
        length=512,
        description="Above 1.0, wider nucleus → more dynamic variation",
    ),
    Preset(
        name="04",
        temperature=1.08,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.24,
        length=512,
        description="Balanced but 2x longer with firmer anti-repeat for structure",
    ),
    Preset(
        name="05",
        temperature=1.08,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.24,
        length=512,
        description="High temp, wide nucleus → chaotic and experimental",
    ),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger.info("Device: %s", device)

    tokenizer = ComposerREMI(params=str(TOKENIZER_PATH))
    model = load_model(CHECKPOINT_PATH, device)

    for preset in PRESETS:
        logger.info(
            "--- Preset: %s | temp=%.2f top_k=%d top_p=%.2f rep_pen=%.2f length=%d ---",
            preset.name,
            preset.temperature,
            preset.top_k,
            preset.top_p,
            preset.repetition_penalty,
            preset.length,
        )
        logger.info("    %s", preset.description)

        token_ids = generate(
            model,
            tokenizer,
            device,
            length=preset.length,
            temperature=preset.temperature,
            top_k=preset.top_k,
            top_p=preset.top_p,
            repetition_penalty=preset.repetition_penalty,
        )

        out_path = OUTPUT_DIR / f"{preset.name}.mid"
        tokens_to_midi_file(tokenizer, token_ids, out_path)

    logger.info("Done. Files saved to %s/", OUTPUT_DIR)
    for p in sorted(OUTPUT_DIR.glob("*.mid")):
        logger.info("  %s  (%d bytes)", p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
