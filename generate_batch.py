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
    top_k: int
    top_p: float
    repetition_penalty: float
    length: int
    description: str


PRESETS: list[Preset] = [
    Preset(
        name="01_liszt",
        composer="Sergei Rachmaninoff",
        temperature=1.02,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.15,
        length=1024,
        description="Liszt: virtuosic, high exploration",
    ),
    Preset(
        name="02_debussy",
        composer="Sergei Rachmaninoff",
        temperature=1.02,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.15,
        length=1024,
        description="Debussy: soft colors, medium free sampling",
    ),
    Preset(
        name="03_schubert",
        composer="Sergei Rachmaninoff",
        temperature=1.02,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.15,
        length=1024,
        description="Schubert: lyrical, tighter anti-repeat",
    ),
    Preset(
        name="04_scarlatti",
        composer="Sergei Rachmaninoff",
        temperature=1.02,   
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.15,
        length=1024,
        description="Scarlatti: crisp, focused baroque keyboard",
    ),
    Preset(
        name="05_scriabin",
        composer="Sergei Rachmaninoff",
        temperature=1.02,
        top_k=60,
        top_p=0.95,
        repetition_penalty=1.15,
        length=1024,
        description="Scriabin: dense harmony, wide nucleus",
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
            "--- Preset: %s | composer=%s temp=%.2f top_k=%d top_p=%.2f length=%d ---",
            preset.name,
            preset.composer,
            preset.temperature,
            preset.top_k,
            preset.top_p,
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
            composer=preset.composer,
            top_p=preset.top_p,
            repetition_penalty=preset.repetition_penalty,
        )

        out_path = OUTPUT_DIR / f"{preset.name}.mid"
        tokens_to_midi_file(tokenizer, token_ids, out_path)

    logger.info("Done. Files saved to %s/", OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.glob("*.mid")):
        logger.info("  %s  (%d bytes)", path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
