"""Generate 5 MIDI files with varied sampling parameters for A/B listening."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from generate import generate, load_model, tokens_to_midi_file
from model import get_device
from scripts.tokenize_midi import ComposerREMI

CHECKPOINT_PATH = Path("checkpoints/model_best.pt")
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
    composer: str
    temperature: float
    min_p: float
    cfg_scale: float
    length: int
    description: str


PRESETS: list[Preset] = [
    Preset(
        name="01_chopin",
        composer="Frédéric Chopin",
        temperature=0.90,
        min_p=0.05,
        cfg_scale=2.0,
        length=1024,
        description="Chopin: default min-p + CFG 2.0",
    ),
    Preset(
        name="02_bach",
        composer="Johann Sebastian Bach",
        temperature=0.80,
        min_p=0.08,
        cfg_scale=2.5,
        length=1024,
        description="Bach: conservative cutoff, strong CFG",
    ),
    Preset(
        name="03_liszt",
        composer="Franz Liszt",
        temperature=1.05,
        min_p=0.03,
        cfg_scale=1.5,
        length=1024,
        description="Liszt: hot sampling, wide tail, milder CFG",
    ),
    Preset(
        name="04_debussy",
        composer="Claude Debussy",
        temperature=0.95,
        min_p=0.05,
        cfg_scale=2.0,
        length=1024,
        description="Debussy: balanced min-p and CFG",
    ),
    Preset(
        name="05_rachmaninoff",
        composer="Sergei Rachmaninoff",
        temperature=1.00,
        min_p=0.10,
        cfg_scale=3.0,
        length=1024,
        description="Rachmaninoff: tight cutoff, strong CFG",
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
            "--- Preset: %s | composer=%s temp=%.2f min_p=%.2f cfg=%.1f ---",
            preset.name,
            preset.composer,
            preset.temperature,
            preset.min_p,
            preset.cfg_scale,
        )
        logger.info("    %s", preset.description)

        token_ids = generate(
            model,
            tokenizer,
            device,
            length=preset.length,
            temperature=preset.temperature,
            composer=preset.composer,
            min_p=preset.min_p,
            cfg_scale=preset.cfg_scale,
        )

        out_path = OUTPUT_DIR / f"{preset.name}.mid"
        tokens_to_midi_file(tokenizer, token_ids, out_path)
        logger.info("Saved %s (%d tokens)", out_path, len(token_ids))

    logger.info("Done. Files saved to %s/", OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.glob("0*.mid")):
        logger.info("  %s  (%d bytes)", path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
