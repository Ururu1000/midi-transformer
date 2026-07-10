from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from miditok import REMI
from torch import nn
from torch.utils.data import DataLoader

from model import MusicTransformer, get_device
from scripts.tokenize_midi import MusicDataset, build_pitch_shift_maps

TOKENS_PATH = Path("data/processed/tokens.pt")
TOKENIZER_PATH = Path("data/processed/tokenizer.json")
CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINT_PATTERN = re.compile(r"model_epoch_(\d+)\.pt$")

# Tuned for an NVIDIA T4 (16GB VRAM): a 27M param model fits batch 128 at
# seq_len 2048, so no gradient accumulation is needed.
BATCH_SIZE = 12
# MPS/CPU have no flash-attention kernel, so SDPA materializes the full
# (batch, heads, seq, seq) score matrix; batch 128 needs ~16GB there. This
# smaller batch keeps local smoke tests on a MacBook within memory.
LOCAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 10
NUM_EPOCHS = 5
WARMUP_EPOCHS = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOG_EVERY = 10
SEED = 42

# DataLoader workers for the Linux VM; pinned memory speeds up host->GPU copies.
NUM_WORKERS = 4

# The RoPE/REMI architecture is incompatible with old absolute-PE checkpoints,
# so training starts from scratch by default.
RESUME_TRAINING = False

# Sequences are tokenized at 2048. Attention memory scales ~O(L^2); set this
# (e.g. 1024) to truncate long sequences and cut VRAM. None keeps full length.
MAX_SEQ_LEN: int | None = None

USE_WANDB = True
WANDB_PROJECT = "ai-music-project"
WANDB_ENTITY: str | None = None
WANDB_RUN_NAME: str | None = None
# "online" | "offline" | "disabled" — offline is useful on air-gapped VMs.
WANDB_MODE = "online"

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)


def get_model_state_dict(model: nn.Module) -> dict[str, Any]:
    """Save clean keys without the torch.compile `_orig_mod.` prefix."""
    return unwrap_model(model).state_dict()


def load_model_state_dict(model: nn.Module, state_dict: dict[str, Any]) -> None:
    """Load weights into a raw or compiled model; strip `_orig_mod.` if present."""
    cleaned = {
        (key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key): value
        for key, value in state_dict.items()
    }
    unwrap_model(model).load_state_dict(cleaned)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    random.setstate(rng_state["python"])
    torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def load_training_data(
    tokens_path: Path,
    tokenizer_path: Path,
) -> tuple[MusicDataset, int, int]:
    checkpoint = torch.load(tokens_path)
    input_ids = checkpoint["input_ids"]
    attention_mask = checkpoint["attention_mask"]
    assert input_ids.ndim == 2, f"Got {input_ids.shape}"
    assert attention_mask.shape == input_ids.shape, (
        f"Got input_ids={input_ids.shape}, attention_mask={attention_mask.shape}"
    )

    vocab_size = int(checkpoint["vocab_size"])
    pad_token_id = int(checkpoint["pad_token_id"])

    tokenizer = REMI(params=str(tokenizer_path))
    pitch_shift_maps = build_pitch_shift_maps(tokenizer, vocab_size)
    dataset = MusicDataset(input_ids.long(), attention_mask.long(), pitch_shift_maps)
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


def save_checkpoint(
    checkpoint_path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    vocab_size: int,
    pad_token_id: int,
    wandb_run_id: str | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": get_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "seed": SEED,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "wandb_run_id": wandb_run_id,
    }
    # Atomic write avoids truncated checkpoints if the VM dies mid-save.
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(checkpoint_path)


def resume_from_checkpoint(
    checkpoint_path: Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, str | None]:
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.warning("No checkpoint found, training from scratch")
        return 0, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    completed_epochs = int(checkpoint["epoch"])
    wandb_run_id = checkpoint.get("wandb_run_id")

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        # Backward-compatible fallback for older checkpoints without scheduler.
        logger.warning(
            "Checkpoint missing scheduler_state_dict; advancing scheduler %d steps",
            completed_epochs,
        )
        for _ in range(completed_epochs):
            scheduler.step()

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    else:
        logger.warning("Checkpoint missing scaler_state_dict; keeping fresh GradScaler")

    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    else:
        logger.warning("Checkpoint missing rng_state; RNG continuity not restored")

    logger.info(
        "Resumed from %s at epoch %d | LR %.2e",
        checkpoint_path,
        completed_epochs,
        scheduler.get_last_lr()[0],
    )
    return completed_epochs, wandb_run_id


def init_wandb(
    *,
    device: torch.device,
    batch_size: int,
    vocab_size: int,
    num_params_m: float,
    dataset_size: int,
    resume_run_id: str | None,
) -> Any:
    if not USE_WANDB or WANDB_MODE == "disabled":
        logger.info("wandb logging disabled")
        return None

    init_kwargs: dict[str, Any] = {
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "name": WANDB_RUN_NAME,
        "mode": WANDB_MODE,
        "config": {
            "batch_size": batch_size,
            "local_batch_size": LOCAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": batch_size * GRADIENT_ACCUMULATION_STEPS,
            "num_epochs": NUM_EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
            "num_workers": NUM_WORKERS,
            "max_seq_len": MAX_SEQ_LEN,
            "resume_training": RESUME_TRAINING,
            "device": str(device),
            "vocab_size": vocab_size,
            "num_params_m": num_params_m,
            "dataset_size": dataset_size,
            "d_model": 768,
            "nhead": 12,
            "num_layers": 16,
            "d_ff": 3072,
        },
    }
    if resume_run_id is not None:
        init_kwargs["id"] = resume_run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)
    logger.info("wandb run: %s (%s)", run.name, run.id)
    return run


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pad_token_id: int,
    epoch: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
    global_step: int,
    use_wandb: bool,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    vocab_size = unwrap_model(model).vocab_size

    optimizer.zero_grad(set_to_none=True)
    for step, (batch, attention_mask) in enumerate(dataloader):
        batch = batch.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        if MAX_SEQ_LEN is not None:
            batch = batch[:, :MAX_SEQ_LEN]
            attention_mask = attention_mask[:, :MAX_SEQ_LEN]

        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        input_mask = attention_mask[:, :-1]
        assert inputs.shape == targets.shape, (
            f"Got inputs={inputs.shape}, targets={targets.shape}"
        )
        assert input_mask.shape == inputs.shape, (
            f"Got input_mask={input_mask.shape}, inputs={inputs.shape}"
        )

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(inputs, attention_mask=input_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=pad_token_id,
            )

        # Scale so accumulated gradients match the mean over the effective batch.
        scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()

        # Step on every Nth batch, and always flush on the final batch so the
        # last (possibly partial) accumulation window is not dropped.
        is_accumulation_boundary = (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
        grad_norm: float | None = None
        if is_accumulation_boundary or (step + 1) == num_batches:
            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=MAX_GRAD_NORM,
                ).item()
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_loss = loss.item()
        total_loss += batch_loss
        global_step += 1

        if (step + 1) % LOG_EVERY == 0:
            logger.info(
                "Epoch %d | Batch %d/%d | Loss %.4f",
                epoch,
                step + 1,
                num_batches,
                batch_loss,
            )
            if use_wandb:
                metrics = {
                    "train/batch_loss": batch_loss,
                    "train/epoch": epoch,
                    "train/batch": step + 1,
                }
                if grad_norm is not None:
                    metrics["train/grad_norm"] = grad_norm
                wandb.log(metrics, step=global_step)

    return total_loss / max(num_batches, 1), global_step


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    set_seed(SEED)
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

    logger.info("Device: %s | AMP dtype: %s | seed=%d", device, amp_dtype, SEED)
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

    # Build optimizer/scheduler on the raw module, resume, then compile so
    # checkpoint keys stay portable and parameter identity is preserved.
    model: nn.Module = MusicTransformer(vocab_size=vocab_size).to(device)
    num_params_m = unwrap_model(model).get_num_params()
    logger.info("Model parameters: %.2fM", num_params_m)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    completed_epochs = 0
    wandb_run_id: str | None = None
    if RESUME_TRAINING:
        latest_checkpoint = find_latest_checkpoint(CHECKPOINTS_DIR)
        completed_epochs, wandb_run_id = resume_from_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
    else:
        logger.info(
            "RESUME_TRAINING is disabled, training from randomly initialized weights"
        )

    wandb_run = init_wandb(
        device=device,
        batch_size=batch_size,
        vocab_size=vocab_size,
        num_params_m=num_params_m,
        dataset_size=len(dataset),
        resume_run_id=wandb_run_id if RESUME_TRAINING else None,
    )
    use_wandb = wandb_run is not None
    if use_wandb:
        wandb_run_id = wandb_run.id

    if device.type == "cuda":
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    start_epoch = completed_epochs + 1
    global_step = completed_epochs * len(dataloader)

    try:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            avg_loss, global_step = train_one_epoch(
                model,
                dataloader,
                optimizer,
                scaler,
                device,
                pad_token_id,
                epoch,
                amp_dtype,
                use_amp,
                global_step,
                use_wandb,
            )
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                "Epoch %d finished | Avg loss %.4f | LR %.2e",
                epoch,
                avg_loss,
                current_lr,
            )
            if use_wandb:
                wandb.log(
                    {
                        "train/epoch_loss": avg_loss,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            checkpoint_path = CHECKPOINTS_DIR / f"model_epoch_{epoch}.pt"
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                vocab_size=vocab_size,
                pad_token_id=pad_token_id,
                wandb_run_id=wandb_run_id,
            )
            logger.info("Saved checkpoint to %s", checkpoint_path)

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()