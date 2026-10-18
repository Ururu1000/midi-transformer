"""Gradio web application for interactive composer-conditioned MIDI generation.

Launch locally (after ``pip install -e '.[app]'``):
    musiclm-app

Deploy to Hugging Face Spaces:
    1. Create a new Space (Gradio SDK) at huggingface.co/new-space
    2. Push this repo and set the app entry point to musiclm.app:main
    3. Upload model_best.pt and tokenizer.json via scripts/upload_to_hf.py.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch

from musiclm.audio import render_midi_to_wav
from musiclm.config import COMPOSER_MAPPING_PATH, TOKENIZER_PATH, GenerateConfig
from musiclm.data.tokenizer import ComposerREMI, list_composer_tokens
from musiclm.inference.prompts import parse_prompt
from musiclm.inference.rerank import load_reference_stats
from musiclm.inference.sampler import generate_piece, load_model, tokens_to_midi_file
from musiclm.model import get_device

DEFAULT_TEMPERATURE = 0.95
DEFAULT_MIN_P = 0.03
DEFAULT_CFG_SCALE = 1.2
DEFAULT_REP_PENALTY = GenerateConfig.repetition_penalty
DEFAULT_LENGTH = 1024
DEFAULT_CANDIDATES = 1
MAX_CANDIDATES = 4
DEFAULT_PRIME_BARS = GenerateConfig.prime_bars
SAMPLE_RATE = 44100
# Share of the progress bar spent on token sampling; MIDI + audio take the rest.
SAMPLING_PROGRESS = 0.85

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppContext:
    """Everything the UI callbacks need, loaded once at startup."""

    device: torch.device
    model: torch.nn.Module
    tokenizer: ComposerREMI
    available_composers: list[str]
    reference_stats: dict[str, Any] | None


def load_resources(
    checkpoint_path: Path = GenerateConfig.checkpoint_path,
    tokenizer_path: Path = TOKENIZER_PATH,
) -> AppContext:
    """Load the model and tokenizer, and build the composer list."""
    device = get_device()
    logger.info("Device: %s", device)

    tokenizer = ComposerREMI(params=str(tokenizer_path))
    logger.info(
        "Loaded tokenizer from %s | vocab_size=%d | bpe=%s",
        tokenizer_path,
        len(tokenizer),
        tokenizer.is_trained,
    )

    model = load_model(checkpoint_path, device)

    # Build composer dropdown from the mapping file or tokenizer vocabulary.
    if COMPOSER_MAPPING_PATH.exists():
        with open(COMPOSER_MAPPING_PATH) as fh:
            mapping = json.load(fh)
        available_composers = list(mapping.get("top_composers", []))
    else:
        available_composers = [
            t.replace("Composer_", "").replace("_", " ")
            for t in list_composer_tokens(tokenizer)
        ]

    logger.info("Available composers: %s", available_composers)
    return AppContext(device=device, model=model, tokenizer=tokenizer,
                      available_composers=available_composers,
                      reference_stats=load_reference_stats())


def midi_to_audio(
    midi_path: Path, sample_rate: int = SAMPLE_RATE
) -> tuple[int, np.ndarray]:
    """Render a MIDI file with the studio grand-piano SoundFont.

    Delegates to :func:`musiclm.audio.render_midi_to_wav` (FluidSynth CLI +
    YDP Grand Piano), so the preview matches ``musiclm-generate --mp3``.
    Returns ``(sample_rate, audio)`` with audio shaped ``(frames, channels)``.
    """
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    wav_path = Path(name)
    try:
        engine = render_midi_to_wav(midi_path, wav_path, sample_rate=sample_rate)
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            rate = wf.getframerate()
            pcm = wf.readframes(wf.getnframes())
    finally:
        wav_path.unlink(missing_ok=True)
    logger.info("Audio preview rendered via %s", engine)

    audio = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).astype(np.float32)
    peak = np.abs(audio).max() if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    return rate, audio


def generate_music(
    ctx: AppContext,
    composer: str,
    temperature: float,
    min_p: float,
    cfg_scale: float,
    repetition_penalty: float,
    length: int,
    progress: gr.Progress | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    primer_path: str | None = None,
    prime_bars: int = DEFAULT_PRIME_BARS,
    keep_primer: bool = False,
) -> tuple[tuple[int, np.ndarray], Path, str]:
    """Generate a new composition; return ((sample_rate, audio), midi_path, summary)."""
    candidates = int(candidates)
    desc = (
        f"Generating {candidates} candidates…" if candidates > 1 else "Generating tokens…"
    )
    if progress is not None:
        progress(0.0, desc=desc)

    def report(done: int, total: int) -> None:
        if progress is not None:
            progress(SAMPLING_PROGRESS * done / max(total, 1), desc=desc)

    result = generate_piece(
        ctx.model,
        ctx.tokenizer,
        ctx.device,
        int(length),
        composer=composer,
        prime_midi=primer_path or None,
        prime_bars=int(prime_bars),
        keep_primer=keep_primer,
        num_candidates=candidates,
        reference_stats=ctx.reference_stats,
        temperature=temperature,
        min_p=min_p,
        cfg_scale=cfg_scale,
        repetition_penalty=repetition_penalty,
        progress_callback=report,
    )
    token_ids = result.token_ids
    logger.info("Generated %d tokens for %s", len(token_ids), composer)

    notes = []
    if result.primer_tokens:
        kept = "kept at the start" if keep_primer else "not included in the output"
        notes.append(f"Primed with the first {int(prime_bars)} bars of your MIDI ({kept}).")
    if result.scores:
        ranked = ", ".join(
            f"**{score:.2f}**" if i == result.chosen else f"{score:.2f}"
            for i, score in enumerate(result.scores)
        )
        notes.append(
            f"Kept candidate {result.chosen + 1} of {len(result.scores)} "
            f"(distance from real {composer}, lower is better: {ranked})."
        )

    if progress is not None:
        progress(SAMPLING_PROGRESS, desc="Converting to MIDI…")

    # delete=False: the file must survive until Gradio serves it; the UI
    # callback deletes it before the next generation.
    fd, name = tempfile.mkstemp(suffix=".mid")
    midi_path = Path(name)
    os.close(fd)
    tokens_to_midi_file(ctx.tokenizer, token_ids, midi_path)

    if progress is not None:
        progress(0.9, desc="Rendering audio preview…")

    return midi_to_audio(midi_path), midi_path, "\n\n".join(notes)


DESCRIPTION = """\
# Classical Music Generator

Generate classical piano compositions using a **LLaMA-style Transformer** \
trained on the MAESTRO and GiantMIDI-Piano datasets.

Select a composer, adjust the sampling parameters, and click **Generate** \
to create a new piece.
"""


def build_ui(ctx: AppContext) -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    )

    # Temp MIDI files are removed at the start of each new generation so a
    # long-running Space does not accumulate them on disk.
    temp_files: list[Path] = []

    def cleanup_temp_files() -> None:
        for path in temp_files:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not remove temp file %s: %s", path, exc)
        temp_files.clear()

    def generate_callback(
        prompt_text: str,
        composer: str,
        temperature: float,
        min_p: float,
        cfg_scale: float,
        repetition_penalty: float,
        length: int,
        candidates: int,
        primer_path: str | None,
        prime_bars: int,
        keep_primer: bool,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple:
        cleanup_temp_files()
        interpretation = ""
        if prompt_text and prompt_text.strip():
            parsed = parse_prompt(prompt_text, ctx.available_composers)
            if parsed.composer:
                composer = parsed.composer
            if parsed.length:
                length = parsed.length
            if parsed.temperature:
                temperature = parsed.temperature
            interpretation = parsed.interpretation
        try:
            (sample_rate, audio), midi_path, summary = generate_music(
                ctx, composer, temperature, min_p, cfg_scale,
                repetition_penalty, length, progress,
                candidates=candidates,
                primer_path=primer_path,
                prime_bars=prime_bars,
                keep_primer=keep_primer,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        temp_files.append(midi_path)
        details = "\n\n".join(part for part in (interpretation, summary) if part)
        return (sample_rate, audio), str(midi_path), details

    with gr.Blocks(theme=theme, title="Music Generator") as app:
        gr.Markdown(DESCRIPTION)

        prompt_box = gr.Textbox(
            lines=2,
            label="Prompt (optional)",
            placeholder="Example: generate Chopin longer",
        )
        interpretation_md = gr.Markdown("")

        with gr.Row():
            # ── Left column: controls ────────────────────────────
            with gr.Column(scale=1):
                composer_dd = gr.Dropdown(
                    choices=ctx.available_composers,
                    value=ctx.available_composers[0] if ctx.available_composers else None,
                    label="Composer",
                    info="Choose a composer style for generation",
                )
                temperature_sl = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=DEFAULT_TEMPERATURE,
                    step=0.05,
                    label="Temperature",
                    info="Higher = more creative & unpredictable",
                )
                min_p_sl = gr.Slider(
                    minimum=0.0,
                    maximum=0.2,
                    value=DEFAULT_MIN_P,
                    step=0.01,
                    label="Min-P",
                    info="Filters tokens below this fraction of the top probability",
                )
                cfg_sl = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=DEFAULT_CFG_SCALE,
                    step=0.1,
                    label="CFG Scale",
                    info="Classifier-Free Guidance strength (1.0 = off)",
                )
                rep_pen_sl = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=DEFAULT_REP_PENALTY,
                    step=0.05,
                    label="Repetition Penalty",
                    info="Pitch repetition suppression (1.0 = off)",
                )
                length_sl = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=DEFAULT_LENGTH,
                    step=128,
                    label="Generation Length (tokens)",
                    info="Number of tokens to generate",
                )
                candidates_sl = gr.Slider(
                    minimum=1,
                    maximum=MAX_CANDIDATES,
                    value=DEFAULT_CANDIDATES,
                    step=1,
                    label="Candidates (best-of-N)",
                    info="Generate several pieces at once and keep the one "
                    "closest to the real composer; slower",
                )
                with gr.Accordion("Start from a real piece (optional)", open=False):
                    primer_file = gr.File(
                        label="Primer MIDI",
                        file_types=[".mid", ".midi"],
                        type="filepath",
                    )
                    prime_bars_sl = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=DEFAULT_PRIME_BARS,
                        step=1,
                        label="Primer bars",
                        info="Opening bars (with notes) fed to the model",
                    )
                    keep_primer_cb = gr.Checkbox(
                        value=False,
                        label="Keep primer bars in the output",
                    )
                generate_btn = gr.Button(
                    "🎹 Generate",
                    variant="primary",
                    size="lg",
                )

            # ── Right column: outputs ────────────────────────────
            with gr.Column(scale=2):
                audio_out = gr.Audio(
                    label="Audio Preview",
                    type="numpy",
                    interactive=False,
                )
                midi_out = gr.File(
                    label="Download MIDI",
                )

        generate_btn.click(
            fn=generate_callback,
            inputs=[
                prompt_box,
                composer_dd,
                temperature_sl,
                min_p_sl,
                cfg_sl,
                rep_pen_sl,
                length_sl,
                candidates_sl,
                primer_file,
                prime_bars_sl,
                keep_primer_cb,
            ],
            outputs=[audio_out, midi_out, interpretation_md],
        )

    return app


def main() -> None:
    ctx = load_resources()
    app = build_ui(ctx)
    app.launch(share=False)


if __name__ == "__main__":
    main()
