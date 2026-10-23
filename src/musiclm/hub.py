"""Fetch released artifacts from the Hugging Face Hub when missing locally.

Only the default locations are resolved this way: a path the user passed
explicitly is never silently swapped for a download. Set ``HF_HUB_OFFLINE=1``
to disable network access; the Hub client then only serves its local cache.
"""
from __future__ import annotations

import logging
from pathlib import Path

from musiclm.config import (
    COMPOSER_MAPPING_PATH,
    HF_REPO_ID,
    REFERENCE_STATS_PATH,
    TOKENIZER_PATH,
    GenerateConfig,
)

# Inference weights, preferred over the full training checkpoint (which also
# carries optimizer state and is several times larger).
HUB_WEIGHTS = "model.safetensors"
HUB_WEIGHTS_CONFIG = "config.json"
HUB_TRAINING_CHECKPOINT = "model_best.pt"

# Default local path -> file name in the Hub repo.
RELEASE_FILES: dict[Path, str] = {
    TOKENIZER_PATH: "tokenizer.json",
    COMPOSER_MAPPING_PATH: "composer_mapping.json",
    REFERENCE_STATS_PATH: "reference_stats.json",
}

logger = logging.getLogger(__name__)


def _download(filename: str, repo_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s from https://huggingface.co/%s", filename, repo_id)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def resolve_release_file(path: Path, repo_id: str = HF_REPO_ID) -> Path:
    """Return *path*, or the released copy when a default file is missing.

    Non-default or present paths come back unchanged so the caller reports
    its own "not found" error.
    """
    path = Path(path)
    filename = RELEASE_FILES.get(path)
    if path.exists() or filename is None:
        return path
    return _download(filename, repo_id)


def resolve_checkpoint(path: Path, repo_id: str = HF_REPO_ID) -> Path:
    """Like :func:`resolve_release_file` for the default model checkpoint."""
    path = Path(path)
    if path.exists() or path != GenerateConfig.checkpoint_path:
        return path

    from huggingface_hub.utils import EntryNotFoundError

    # Probing by download (not by listing the repo) keeps HF_HUB_OFFLINE
    # working from the cache. config.json lands next to the weights.
    try:
        _download(HUB_WEIGHTS_CONFIG, repo_id)
    except EntryNotFoundError:
        return _download(HUB_TRAINING_CHECKPOINT, repo_id)
    return _download(HUB_WEIGHTS, repo_id)
