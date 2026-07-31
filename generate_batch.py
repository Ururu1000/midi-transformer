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
    repetition_penalty: float
    length: int
    description: str


PRESETS: list[Preset] = [
    Preset(
        name="01_chopin",
        composer="Frédéric Chopin",
        temperature=0.70,
        min_p=0.08,
        cfg_scale=1.0,
        repetition_penalty=1.20,
        length=1536,
        description="Chopin: baseline, CFG off, repetition penalty 1.2",
    ),
    Preset(
        name="02_bach",
        composer="Johann Sebastian Bach",
        temperature=0.70,
        min_p=0.10,
        cfg_scale=1.0,
        repetition_penalty=1.30,
        length=1536,
        description="Bach: tighter cutoff, harder anti-loop penalty",
    ),
    Preset(
        name="03_liszt",
        composer="Franz Liszt",
        temperature=0.85,
        min_p=0.06,
        cfg_scale=1.0,
        repetition_penalty=1.15,
        length=1536,
        description="Liszt: hotter sampling, lighter penalty",
    ),
    Preset(
        name="04_debussy",
        composer="Claude Debussy",
        temperature=0.75,
        min_p=0.08,
        cfg_scale=1.0,
        repetition_penalty=1.20,
        length=1536,
        description="Debussy: balanced cool sampling",
    ),
    Preset(
        name="05_rachmaninoff",
        composer="Sergei Rachmaninoff",
        temperature=0.75,
        min_p=0.10,
        cfg_scale=1.2,
        repetition_penalty=1.25,
        length=1536,
        description="Rachmaninoff: slight CFG, strong penalty",
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
            "--- Preset: %s | composer=%s temp=%.2f min_p=%.2f cfg=%.1f rep=%.2f ---",
            preset.name,
            preset.composer,
            preset.temperature,
            preset.min_p,
            preset.cfg_scale,
            preset.repetition_penalty,
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
            repetition_penalty=preset.repetition_penalty,
        )

        out_path = OUTPUT_DIR / f"{preset.name}.mid"
        tokens_to_midi_file(tokenizer, token_ids, out_path)
        logger.info("Saved %s (%d tokens)", out_path, len(token_ids))

    logger.info("Done. Files saved to %s/", OUTPUT_DIR)
    for path in sorted(OUTPUT_DIR.glob("0*.mid")):
        logger.info("  %s  (%d bytes)", path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
