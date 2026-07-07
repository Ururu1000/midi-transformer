from __future__ import annotations

import logging
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from miditok import REMI
from torch.utils.data import DataLoader

from model import MusicTransformer, get_device
from scripts.tokenize_midi import MusicDataset, build_pitch_shift_maps

TOKENS_PATH = Path("data/processed/tokens.pt")
TOKENIZER_PATH = Path("data/processed/tokenizer.json")
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINT_PATTERN = re.compile(r"model_epoch_(\d+)\.pt$")

# Tuned for an NVIDIA T4 (16GB VRAM): a 27M param model fits batch 128 at
# seq_len 2048, so no gradient accumulation is needed.
BATCH_SIZE = 128
# MPS/CPU have no flash-attention kernel, so SDPA materializes the full
# (batch, heads, seq, seq) score matrix; batch 128 needs ~16GB there. This
# smaller batch keeps local smoke tests on a MacBook within memory.
LOCAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
LOG_EVERY = 10

# DataLoader workers for the Linux VM; pinned memory speeds up host->GPU copies.
NUM_WORKERS = 4

# The RoPE/REMI architecture is incompatible with old absolute-PE checkpoints,
# so training starts from scratch by default.
RESUME_TRAINING = False

# Sequences are tokenized at 2048. Attention memory scales ~O(L^2); set this
# (e.g. 1024) to truncate long sequences and cut VRAM. None keeps full length.
MAX_SEQ_LEN: int | None = None

logger = logging.getLogger(__name__)


def load_training_data(
    tokens_path: Path,
    tokenizer_path: Path,
) -> tuple[MusicDataset, int, int]:
    checkpoint = torch.load(tokens_path)
    input_ids = checkpoint["input_ids"]
    assert input_ids.ndim == 2, f"Got {input_ids.shape}"

    vocab_size = int(checkpoint["vocab_size"])
    pad_token_id = int(checkpoint["pad_token_id"])

    tokenizer = REMI(params=str(tokenizer_path))
    pitch_shift_maps = build_pitch_shift_maps(tokenizer, vocab_size)
    dataset = MusicDataset(input_ids.long(), pitch_shift_maps)
    return dataset, vocab_size, pad_token_id


def find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    latest_epoch = -1
    latest_path: Path | None = None
    for path in checkpoints_dir.glob("model_epoch_*.pt"):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_path = path
    return latest_path


def resume_from_checkpoint(
    checkpoint_path: Path | None,
    model: MusicTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.warning("No checkpoint found, training from scratch")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epochs = int(checkpoint["epoch"])
    logger.info("Resumed from %s at epoch %d", checkpoint_path, completed_epochs)
    return completed_epochs


def train_one_epoch(
    model: MusicTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pad_token_id: int,
    epoch: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(dataloader):
        batch = batch.to(device, non_blocking=True)
        if MAX_SEQ_LEN is not None:
            batch = batch[:, :MAX_SEQ_LEN]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        assert inputs.shape == targets.shape, (
            f"Got inputs={inputs.shape}, targets={targets.shape}"
        )

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                ignore_index=pad_token_id,
            )

        # Scale so accumulated gradients match the mean over the effective batch.
        scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()

        # Step on every Nth batch, and always flush on the final batch so the
        # last (possibly partial) accumulation window is not dropped.
        is_accumulation_boundary = (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
        if is_accumulation_boundary or (step + 1) == num_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        if (step + 1) % LOG_EVERY == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Loss %.4f",
                epoch,
                step + 1,
                num_batches,
                loss.item(),
            )

    return total_loss / max(num_batches, 1)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = get_device()

    # bfloat16 is used on Ampere+; older CUDA GPUs (e.g. T4) fall back to fp16,
    # which needs a GradScaler. On CPU mixed precision is disabled entirely.
    use_amp = device.type == "cuda"
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    batch_size = BATCH_SIZE if device.type == "cuda" else LOCAL_BATCH_SIZE

    logger.info("Device: %s | AMP dtype: %s", device, amp_dtype)
    logger.info(
        "Batch size %d x %d accumulation steps = effective batch %d",
        batch_size,
        GRADIENT_ACCUMULATION_STEPS,
        batch_size * GRADIENT_ACCUMULATION_STEPS,
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset, vocab_size, pad_token_id = load_training_data(TOKENS_PATH, TOKENIZER_PATH)
    logger.info(
        "Loaded %d sequences | vocab_size=%d | pad_token_id=%d",
        len(dataset),
        vocab_size,
        pad_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = MusicTransformer(vocab_size=vocab_size).to(device)
    logger.info("Model parameters: %.2fM", model.get_num_params())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    completed_epochs = 0
    if RESUME_TRAINING:
        latest_checkpoint = find_latest_checkpoint(CHECKPOINTS_DIR)
        completed_epochs = resume_from_checkpoint(
            latest_checkpoint, model, optimizer, device
        )
        for _ in range(completed_epochs):
            scheduler.step()
    else:
        logger.info("RESUME_TRAINING is disabled, training from randomly initialized weights")
    start_epoch = completed_epochs + 1

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            pad_token_id,
            epoch,
            amp_dtype,
            use_amp,
        )
        scheduler.step()

        logger.info(
            "Epoch %d finished | Avg loss %.4f | LR %.2e",
            epoch,
            avg_loss,
            scheduler.get_last_lr()[0],
        )

        checkpoint_path = CHECKPOINTS_DIR / f"model_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "pad_token_id": pad_token_id,
            },
            checkpoint_path,
        )
        logger.info("Saved checkpoint to %s", checkpoint_path)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
