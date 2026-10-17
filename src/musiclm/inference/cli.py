"""CLI entry point for single-file generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from musiclm.config import GenerateConfig
from musiclm.inference.sampler import main as run_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="musiclm-generate",
        description="Generate a composer-conditioned MIDI file.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--composer", type=str, default=None, help='e.g. "Frédéric Chopin"'
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help='Free-text request, e.g. "сгенерируй мне шопена, подлиннее". '
        "Parsed into composer/length/temperature; explicit flags win.",
    )
    parser.add_argument("--length", type=int, default=None, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument(
        "--cfg-scale", type=float, default=None, help="1.0 disables guidance"
    )
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--penalty-window", type=int, default=None)
    parser.add_argument(
        "--seed", type=int, default=None, help="Set for reproducible sampling"
    )
    parser.add_argument(
        "--prime-midi",
        type=Path,
        default=None,
        help="Seed generation with the opening bars of this MIDI file",
    )
    parser.add_argument(
        "--prime-bars", type=int, default=None, help="Bars of --prime-midi to use"
    )
    parser.add_argument(
        "--keep-primer",
        action="store_true",
        help="Keep the primer bars at the start of the output",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="Best-of-N: sample N pieces in one batch, keep the most realistic",
    )
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Also render an MP3 next to the MIDI (MIDI is always kept)",
    )
    parser.add_argument("--bitrate", type=str, default="192k", help="MP3 bitrate")
    parser.add_argument("--soundfont", type=Path, default=None, help="Path to a .sf2 file")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    defaults = GenerateConfig()
    parsed = None
    if args.prompt:
        from musiclm.inference.prompts import load_available_composers, parse_prompt

        parsed = parse_prompt(
            args.prompt,
            load_available_composers(tokenizer_path=args.tokenizer),
        )
        print(parsed.interpretation)
    cfg = GenerateConfig(
        checkpoint_path=args.checkpoint or defaults.checkpoint_path,
        tokenizer_path=args.tokenizer or defaults.tokenizer_path,
        output_path=args.output or defaults.output_path,
        composer=args.composer
        if args.composer is not None
        else (parsed.composer if parsed and parsed.composer else defaults.composer),
        length=args.length
        if args.length is not None
        else (parsed.length if parsed and parsed.length else defaults.length),
        temperature=(
            args.temperature
            if args.temperature is not None
            else (
                parsed.temperature
                if parsed and parsed.temperature
                else defaults.temperature
            )
        ),
        min_p=args.min_p if args.min_p is not None else defaults.min_p,
        cfg_scale=args.cfg_scale if args.cfg_scale is not None else defaults.cfg_scale,
        repetition_penalty=(
            args.repetition_penalty
            if args.repetition_penalty is not None
            else defaults.repetition_penalty
        ),
        penalty_window=(
            args.penalty_window if args.penalty_window is not None
            else defaults.penalty_window
        ),
        seed=args.seed,
        prime_midi=args.prime_midi,
        prime_bars=args.prime_bars if args.prime_bars is not None else defaults.prime_bars,
        keep_primer=args.keep_primer,
        num_candidates=(
            args.candidates if args.candidates is not None else defaults.num_candidates
        ),
    )
    midi_path = run_generation(cfg)

    if args.mp3:
        from musiclm.audio import convert_midi_to_mp3

        convert_midi_to_mp3(
            midi_path,
            bitrate=args.bitrate,
            soundfont=args.soundfont,
        )


if __name__ == "__main__":
    main()
