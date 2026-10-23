"""Checkpoint I/O: training checkpoints, exported inference weights, model config.

Two on-disk formats load into the same model:

* training checkpoints (``*.pt``) written by the trainer, with optimizer,
  scheduler and RNG state next to the weights;
* inference exports (``model.safetensors`` + ``config.json``) written by
  ``musiclm-export``: weights only, a fraction of the size, no pickle.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from musiclm.config import EXPORT_DIR, GenerateConfig, ModelConfig

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
EXPORT_FORMAT = "musiclm-weights-v1"
EXPORT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
# token_embedding and lm_head share one tensor; safetensors refuses aliased
# tensors, so exports store lm_head only and the tie is restored on load.
TIED_KEY = "token_embedding.weight"
TIED_SOURCE_KEY = "lm_head.weight"

logger = logging.getLogger(__name__)


def strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``_orig_mod.`` prefix torch.compile adds to parameter names."""
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def model_config_from_dict(payload: dict[str, Any]) -> ModelConfig:
    known = {field.name for field in fields(ModelConfig)}
    return ModelConfig(**{key: value for key, value in payload.items() if key in known})


def load_weights(path: Path) -> tuple[dict[str, Tensor], ModelConfig]:
    """State dict and architecture from a training checkpoint or an export."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
        state_dict[TIED_KEY] = state_dict[TIED_SOURCE_KEY]
        payload = json.loads(path.with_name(CONFIG_NAME).read_text(encoding="utf-8"))
        return state_dict, model_config_from_dict(payload["model_config"])

    # Training checkpoints also carry ~2x the weights in optimizer state; load
    # on CPU so only the model weights ever reach the accelerator. The file may
    # come from the Hub, so never unpickle arbitrary objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
    # Checkpoints saved before model_config was stored all use the default
    # architecture, so their vocab size is the only free parameter.
    config_payload = checkpoint.get("model_config") or {
        "vocab_size": int(checkpoint["vocab_size"])
    }
    return state_dict, model_config_from_dict(config_payload)


def export_inference_weights(
    checkpoint_path: Path,
    output_dir: Path,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Write ``model.safetensors`` + ``config.json``; returns the weights path."""
    from safetensors.torch import save_file

    state_dict, config = load_weights(checkpoint_path)
    tensors = {
        key: value.to(dtype).contiguous()
        for key, value in state_dict.items()
        if key != TIED_KEY
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / WEIGHTS_NAME
    save_file(tensors, str(weights_path), metadata={"format": "pt"})
    payload = {
        "format": EXPORT_FORMAT,
        "dtype": str(dtype).removeprefix("torch."),
        "source_checkpoint": Path(checkpoint_path).name,
        "model_config": asdict(config),
    }
    (output_dir / CONFIG_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return weights_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        prog="musiclm-export",
        description="Export inference-only weights (safetensors + config.json) "
        "from a training checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, default=GenerateConfig.checkpoint_path)
    parser.add_argument("--output-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--dtype", choices=sorted(EXPORT_DTYPES), default="bfloat16")
    args = parser.parse_args(argv)

    weights_path = export_inference_weights(
        args.checkpoint, args.output_dir, EXPORT_DTYPES[args.dtype]
    )
    logger.info(
        "Wrote %s (%.0f MB) and %s",
        weights_path,
        weights_path.stat().st_size / 1e6,
        weights_path.with_name(CONFIG_NAME),
    )


if __name__ == "__main__":
    main()
