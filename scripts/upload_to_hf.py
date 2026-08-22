"""Upload model checkpoint, tokenizer and metadata to Hugging Face Hub."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from musiclm.config import (
    CHECKPOINTS_DIR,
    COMPOSER_MAPPING_PATH,
    DEFAULT_CHECKPOINT_PATH,
    PROJECT_ROOT,
    TOKENIZER_PATH,
)

HF_USERNAME = "sagevoice"
REPO_NAME = "music-transformer-classical"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Checkpoint to publish (default: checkpoints/model_best.pt)",
    )
    parser.add_argument("--repo", type=str, default=REPO_ID)
    args = parser.parse_args(argv)

    if not HF_USERNAME or HF_USERNAME == "YOUR_HF_USERNAME":
        logger.error("Please set HF_USERNAME in scripts/upload_to_hf.py.")
        return

    files_to_upload: list[tuple[Path, str]] = [
        (args.checkpoint, args.checkpoint.name),
        (TOKENIZER_PATH, "tokenizer.json"),
        (COMPOSER_MAPPING_PATH, "composer_mapping.json"),
        (PROJECT_ROOT / "README.md", "README.md"),
        (PROJECT_ROOT / "LICENSE", "LICENSE"),
    ]

    api = HfApi()

    logger.info("Creating repository %s ...", args.repo)
    create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)

    for local_path, repo_path in files_to_upload:
        if not local_path.exists():
            logger.warning("File not found, skipping: %s", local_path)
            continue
        size_mb = local_path.stat().st_size / 1e6
        logger.info("Uploading %s -> %s (%.1f MB) ...", local_path, repo_path, size_mb)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=args.repo,
        )

    logger.info("Done! Model published at: https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
