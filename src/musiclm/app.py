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
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import pretty_midi
import torch

from musiclm.config import COMPOSER_MAPPING_PATH, TOKENIZER_PATH, GenerateConfig
from musiclm.data.tokenizer import ComposerREMI, list_composer_tokens
from musiclm.inference.sampler import generate, load_model, tokens_to_midi_file
from musiclm.model import get_device

DEFAULT_TEMPERATURE = 0.95
DEFAULT_MIN_P = 0.03
DEFAULT_CFG_SCALE = 1.2
DEFAULT_REP_PENALTY = 1.0
DEFAULT_LENGTH = 1024
SAMPLE_RATE = 44100

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
                      available_composers=available_composers)


def midi_to_audio(midi_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render a MIDI file to a mono float32 waveform.

    Uses ``pretty_midi.fluidsynth()`` when FluidSynth + a SoundFont are
    installed; otherwise falls back to ``synthesize()``, which produces
    simple sine waves — good enough for a quick preview.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    try:
        audio = pm.fluidsynth(fs=sample_rate)
    except Exception:
        logger.info("FluidSynth unavailable, falling back to sine synthesis")
        audio = pm.synthesize(fs=sample_rate)

    # Normalise to [-1, 1] so Gradio can play it without clipping.
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    return audio.astype(np.float32)


def generate_music(
    ctx: AppContext,
    composer: str,
    temperature: float,
    min_p: float,
    cfg_scale: float,
    repetition_penalty: float,
    length: int,
    progress: gr.Progress | None = None,
) -> tuple[tuple[int, np.ndarray], Path]:
    """Generate a new composition; return ((sample_rate, audio), midi_path)."""
    if progress is not None:
        progress(0.0, desc="Generating tokens…")

    token_ids = generate(
        ctx.model,
        ctx.tokenizer,
        ctx.device,
        length=int(length),
        temperature=temperature,
        composer=composer,
        min_p=min_p,
        cfg_scale=cfg_scale,
        repetition_penalty=repetition_penalty,
    )
    logger.info("Generated %d tokens for %s", len(token_ids), composer)

    if progress is not None:
        progress(0.7, desc="Converting to MIDI…")

    # delete=False: the file must survive until Gradio serves it; the UI
    # callback deletes it before the next generation.
    fd, name = tempfile.mkstemp(suffix=".mid")
    midi_path = Path(name)
    os.close(fd)
    tokens_to_midi_file(ctx.tokenizer, token_ids, midi_path)

    if progress is not None:
        progress(0.9, desc="Rendering audio preview…")

    audio = midi_to_audio(midi_path)

    return (SAMPLE_RATE, audio), midi_path


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
        composer: str,
        temperature: float,
        min_p: float,
        cfg_scale: float,
        repetition_penalty: float,
        length: int,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple:
        cleanup_temp_files()
        (sample_rate, audio), midi_path = generate_music(
            ctx, composer, temperature, min_p, cfg_scale,
            repetition_penalty, length, progress,
        )
        temp_files.append(midi_path)
        return (sample_rate, audio), str(midi_path)

    with gr.Blocks(theme=theme, title="Music Generator") as app:
        gr.Markdown(DESCRIPTION)

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
                composer_dd,
                temperature_sl,
                min_p_sl,
                cfg_sl,
                rep_pen_sl,
                length_sl,
            ],
            outputs=[audio_out, midi_out],
        )

    return app


def main() -> None:
    ctx = load_resources()
    app = build_ui(ctx)
    app.launch(share=False)


if __name__ == "__main__":
    main()
