"""Upload model weights, tokenizer and metadata to Hugging Face Hub."""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from musiclm.checkpoint import CONFIG_NAME, export_inference_weights
from musiclm.config import (
    COMPOSER_MAPPING_PATH,
    HF_REPO_ID,
    PROJECT_ROOT,
    REFERENCE_STATS_PATH,
    TOKENIZER_PATH,
    GenerateConfig,
)
from musiclm.hub import HUB_TRAINING_CHECKPOINT, HUB_WEIGHTS, HUB_WEIGHTS_CONFIG

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=GenerateConfig.checkpoint_path,
        help="Checkpoint to publish (default: the one inference uses, "
        "GenerateConfig.checkpoint_path)",
    )
    parser.add_argument("--repo", type=str, default=HF_REPO_ID)
    parser.add_argument(
        "--include-training-state",
        action="store_true",
        help="Also upload the full training checkpoint as model_best.pt "
        "(optimizer state included, several times larger than the export)",
    )
    args = parser.parse_args(argv)

    api = HfApi()
    logger.info("Creating repository %s ...", args.repo)
    create_repo(repo_id=args.repo, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = export_inference_weights(args.checkpoint, Path(tmp))
        files_to_upload: list[tuple[Path, str]] = [
            (weights_path, HUB_WEIGHTS),
            (weights_path.with_name(CONFIG_NAME), HUB_WEIGHTS_CONFIG),
            (TOKENIZER_PATH, "tokenizer.json"),
            (COMPOSER_MAPPING_PATH, "composer_mapping.json"),
            (REFERENCE_STATS_PATH, "reference_stats.json"),
            (PROJECT_ROOT / "README.md", "README.md"),
            (PROJECT_ROOT / "LICENSE", "LICENSE"),
        ]
        if args.include_training_state:
            # The Hub name stays fixed so downloads keep working whatever the
            # local checkpoint is called.
            files_to_upload.append((args.checkpoint, HUB_TRAINING_CHECKPOINT))

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
