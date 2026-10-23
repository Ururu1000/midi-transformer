"""Preset runner: one MIDI per composer preset for A/B listening."""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from musiclm.config import GENERATED_DIR, GenerateConfig
from musiclm.inference.rerank import load_reference_stats
from musiclm.inference.sampler import (
    generate_piece,
    load_model,
    load_tokenizer,
    tokens_to_midi_file,
)
from musiclm.model import get_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    name: str
    composer: str
    # Sampling defaults come from GenerateConfig; presets override per composer.
    temperature: float = GenerateConfig.temperature
    min_p: float = GenerateConfig.min_p
    cfg_scale: float = GenerateConfig.cfg_scale
    repetition_penalty: float = GenerateConfig.repetition_penalty
    length: int = 2048


PRESETS: list[Preset] = [
    Preset(
        name="01_chopin",
        composer="Frédéric Chopin",
        length=4096,
    ),
    Preset(name="02_bach", composer="Johann Sebastian Bach"),
    Preset(name="03_beethoven", composer="Ludwig van Beethoven"),
    Preset(
        name="04_debussy",
        composer="Claude Debussy",
        repetition_penalty=1.3,
    ),
    Preset(
        name="05_rachmaninoff",
        composer="Sergei Rachmaninoff",
        repetition_penalty=1.2,
    ),
]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="musiclm-batch",
        description="Generate one MIDI per preset for A/B listening.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Base seed; each preset adds its index")
    parser.add_argument(
        "--candidates",
        type=int,
        default=GenerateConfig.num_candidates,
        help="Best-of-N per preset: sample N pieces, keep the most realistic",
    )
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Also render an MP3 next to each MIDI (MIDI is always kept)",
    )
    parser.add_argument("--bitrate", type=str, default="192k", help="MP3 bitrate")
    args = parser.parse_args(argv)

    defaults = GenerateConfig()
    checkpoint_path = args.checkpoint or defaults.checkpoint_path
    tokenizer_path = args.tokenizer or defaults.tokenizer_path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger.info("Device: %s", device)

    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(checkpoint_path, device)
    reference_stats = load_reference_stats() if args.candidates > 1 else None

    for index, preset in enumerate(PRESETS):
        logger.info(
            "--- Preset: %s | composer=%s temp=%.2f min_p=%.2f cfg=%.1f rep=%.2f ---",
            preset.name,
            preset.composer,
            preset.temperature,
            preset.min_p,
            preset.cfg_scale,
            preset.repetition_penalty,
        )

        token_ids = generate_piece(
            model,
            tokenizer,
            device,
            preset.length,
            composer=preset.composer,
            num_candidates=args.candidates,
            reference_stats=reference_stats,
            temperature=preset.temperature,
            min_p=preset.min_p,
            cfg_scale=preset.cfg_scale,
            repetition_penalty=preset.repetition_penalty,
            seed=None if args.seed is None else args.seed + index,
        ).token_ids

        out_path = args.output_dir / f"{preset.name}.mid"
        tokens_to_midi_file(tokenizer, token_ids, out_path)
        logger.info("Saved %s (%d tokens)", out_path, len(token_ids))

        if args.mp3:
            from musiclm.audio import convert_midi_to_mp3

            convert_midi_to_mp3(out_path, bitrate=args.bitrate)

    logger.info("Done. Files saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
