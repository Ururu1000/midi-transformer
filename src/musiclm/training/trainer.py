"""Training loop: AMP, gradient accumulation, W&B, early stopping, checkpoints."""
from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader

from musiclm.checkpoint import strip_compile_prefix
from musiclm.config import (
    CHECKPOINTS_DIR,
    TOKENIZER_PATH,
    TRAIN_TOKENS_PATH,
    VAL_TOKENS_PATH,
    ModelConfig,
    TrainConfig,
    model_config_dict,
)
from musiclm.data.dataset import PackedMusicDataset
from musiclm.data.preprocess import DOCS_FORMAT
from musiclm.data.tokenizer import (
    POSITIONS_PER_BEAT,
    UNCONDITIONAL_COMPOSER,
    ComposerREMI,
    composer_vocab_token,
    learned_token_id,
)
from musiclm.model import MusicTransformer, get_device

CHECKPOINT_PATTERN = re.compile(r"model_epoch_(\d+)\.pt$")
# Checkpoints record which LR schedule produced their scheduler state; older
# ones stepped once per epoch and cannot be loaded into the per-step schedule.
SCHEDULER_KIND = "per_step_warmup_cosine"
# Warmup starts at this fraction of the peak learning rate.
WARMUP_START_FACTOR = 0.1

# L4 24GB: seq 4096 attention is ~4x seq 2048. The block-diagonal attn_mask
# routes SDPA to the memory-efficient backend (flash kernels reject arbitrary
# masks); drop batch size further if the GPU OOMs.
EXPECTED_BEAT_RES = {(0, 4): POSITIONS_PER_BEAT, (4, 12): POSITIONS_PER_BEAT}

logger = logging.getLogger(__name__)


def assert_tokenizer_matches_training_config(tokenizer: ComposerREMI) -> None:
    """Refuse stale tokenizer/tokens from before rests/chords/beat_res=12."""
    config = tokenizer.config
    beat_res = dict(config.beat_res)
    problems: list[str] = []
    if beat_res != EXPECTED_BEAT_RES:
        problems.append(f"beat_res={beat_res} expected={EXPECTED_BEAT_RES}")
    if not config.use_rests:
        problems.append("use_rests=False expected True")
    if not config.use_chords:
        problems.append("use_chords=False expected True")
    rest_count = sum(1 for token in tokenizer.vocab if token.startswith("Rest_"))
    chord_count = sum(1 for token in tokenizer.vocab if token.startswith("Chord_"))
    if rest_count == 0:
        problems.append("no Rest_* tokens in vocab")
    if chord_count == 0:
        problems.append("no Chord_* tokens in vocab")

    if problems:
        raise ValueError(
            "Stale tokenizer/tokens detected. Re-run "
            "`musiclm-tokenize` before training. "
            + "; ".join(problems)
        )


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
    unwrap_model(model).load_state_dict(strip_compile_prefix(state_dict))


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(state: Any) -> Tensor:
    # torch.load(..., map_location="cuda") remaps nested RNG tensors onto the
    # GPU; both the CPU and CUDA generators only accept CPU ByteTensors.
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"Expected RNG Tensor, got {type(state)}")
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    random.setstate(rng_state["python"])
    torch.set_rng_state(_as_cpu_byte_tensor(rng_state["torch"]))
    if torch.cuda.is_available() and rng_state.get("cuda") is not None:
        cuda_states = rng_state["cuda"]
        if isinstance(cuda_states, torch.Tensor):
            cuda_states = [cuda_states]
        torch.cuda.set_rng_state_all(
            [_as_cpu_byte_tensor(state) for state in cuda_states]
        )


def load_docs_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tokenized documents not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != DOCS_FORMAT:
        raise ValueError(
            f"{path} has format={checkpoint.get('format')!r}, expected "
            f"{DOCS_FORMAT!r}. Re-run `musiclm-tokenize`."
        )
    return checkpoint


def load_datasets(
    cfg: TrainConfig,
    train_tokens_path: Path,
    val_tokens_path: Path,
    tokenizer_path: Path,
) -> tuple[PackedMusicDataset, PackedMusicDataset, int, int]:
    train_checkpoint = load_docs_checkpoint(train_tokens_path)
    val_checkpoint = load_docs_checkpoint(val_tokens_path)

    vocab_size = int(train_checkpoint["vocab_size"])
    pad_token_id = int(train_checkpoint["pad_token_id"])
    if int(val_checkpoint["vocab_size"]) != vocab_size:
        raise ValueError("Train and validation vocab_size values do not match")
    if int(val_checkpoint["pad_token_id"]) != pad_token_id:
        raise ValueError("Train and validation pad_token_id values do not match")

    tokenizer = ComposerREMI(params=str(tokenizer_path))
    if not tokenizer.is_trained:
        raise ValueError(
            "Tokenizer has no trained BPE model. Re-run `musiclm-tokenize`."
        )
    if len(tokenizer) != vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={len(tokenizer)} != data vocab_size={vocab_size}"
        )
    assert_tokenizer_matches_training_config(tokenizer)

    uncond_id = learned_token_id(
        tokenizer, composer_vocab_token(UNCONDITIONAL_COMPOSER)
    )
    logger.info(
        "Composer conditioning: uncond_id=%d | cfg_drop_prob=%.2f",
        uncond_id,
        cfg.cfg_drop_prob,
    )

    train_dataset = PackedMusicDataset(
        train_checkpoint["docs"],
        cfg.pack_seq_len,
        pad_token_id,
        uncond_id,
        cfg.cfg_drop_prob,
        shifts=list(train_checkpoint["shifts"]),
        augment=True,
    )
    val_dataset = PackedMusicDataset(
        val_checkpoint["docs"],
        cfg.pack_seq_len,
        pad_token_id,
        uncond_id,
        cfg_drop_prob=0.0,
        shifts=list(val_checkpoint["shifts"]),
        augment=False,
    )
    return train_dataset, val_dataset, vocab_size, pad_token_id


def list_epoch_checkpoints(checkpoints_dir: Path) -> list[tuple[int, Path]]:
    """(epoch, path) of every ``model_epoch_*.pt``, oldest first."""
    found: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("model_epoch_*.pt"):
        match = CHECKPOINT_PATTERN.search(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    return sorted(found)


def find_latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    checkpoints = list_epoch_checkpoints(checkpoints_dir)
    return checkpoints[-1][1] if checkpoints else None


def prune_epoch_checkpoints(checkpoints_dir: Path, keep: int) -> None:
    """Delete all but the newest *keep* epoch checkpoints; ``keep < 1`` keeps all."""
    if keep < 1:
        return
    for _, path in list_epoch_checkpoints(checkpoints_dir)[:-keep]:
        path.unlink()
        logger.info("Removed old checkpoint %s", path)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decay weight matrices only; RMSNorm gains and biases (ndim < 2) are exempt."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    # parameters() yields the tied embedding/lm_head tensor once.
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_multiplier(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float
) -> float:
    """Linear warmup to 1.0, then cosine decay to *min_lr_ratio*."""
    if step < warmup_steps:
        return WARMUP_START_FACTOR + (1.0 - WARMUP_START_FACTOR) * step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-optimizer-step schedule; epochs only set the warmup and total lengths."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.num_epochs * steps_per_epoch
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(step, warmup_steps, total_steps, cfg.min_lr_ratio),
    )


def save_checkpoint(
    checkpoint_path: Path,
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    model_config: ModelConfig,
    seed: int,
    pad_token_id: int,
    best_val_loss: float,
    wandb_run_id: str | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": get_model_state_dict(model),
        "model_config": asdict(model_config),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler_kind": SCHEDULER_KIND,
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "seed": seed,
        "vocab_size": model_config.vocab_size,
        "pad_token_id": pad_token_id,
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb_run_id,
    }
    # Atomic write avoids truncated checkpoints if the VM dies mid-save.
    temp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(checkpoint_path)


def _read_checkpoint_counters(
    checkpoint: dict[str, Any],
    batches_per_epoch: int,
) -> tuple[int, int, float, str | None]:
    completed_epochs = int(checkpoint["epoch"])
    global_step = int(
        checkpoint.get("global_step", completed_epochs * batches_per_epoch)
    )
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    wandb_run_id = checkpoint.get("wandb_run_id")
    return completed_epochs, global_step, best_val_loss, wandb_run_id


def resume_weights_only(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
    batches_per_epoch: int,
    cfg: TrainConfig,
) -> tuple[int, int, float, str | None]:
    """Load model weights and loop counters; optimizer must stay fresh."""
    if not checkpoint_path.exists():
        logger.warning(
            "Resume checkpoint %s not found, training from scratch", checkpoint_path
        )
        return 0, 0, float("inf"), None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])

    completed_epochs, global_step, best_val_loss, wandb_run_id = _read_checkpoint_counters(
        checkpoint, batches_per_epoch
    )

    logger.info(
        "Resumed weights only from %s at epoch %d step %d | best val %.4f | "
        "optimizer/scaler fresh (dropout=%.2f weight_decay=%.2f)",
        checkpoint_path,
        completed_epochs,
        global_step,
        best_val_loss,
        cfg.dropout,
        cfg.weight_decay,
    )
    return completed_epochs, global_step, best_val_loss, wandb_run_id


def resume_from_checkpoint(
    checkpoint_path: Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batches_per_epoch: int,
    steps_per_epoch: int,
) -> tuple[int, int, float, str | None]:
    if checkpoint_path is None or not checkpoint_path.exists():
        logger.warning("No checkpoint found, training from scratch")
        return 0, 0, float("inf"), None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except ValueError as exc:
        # Checkpoints from before the decay/no-decay split have one param group.
        logger.warning("Optimizer state not restored (%s); keeping fresh AdamW", exc)

    completed_epochs, global_step, best_val_loss, wandb_run_id = _read_checkpoint_counters(
        checkpoint, batches_per_epoch
    )

    if checkpoint.get("scheduler_kind") == SCHEDULER_KIND:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        completed_steps = completed_epochs * steps_per_epoch
        logger.warning(
            "Checkpoint has no per-step scheduler state; advancing scheduler %d steps",
            completed_steps,
        )
        advance_scheduler(scheduler, completed_steps)

    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    else:
        logger.warning("Checkpoint missing scaler_state_dict; keeping fresh GradScaler")

    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    else:
        logger.warning("Checkpoint missing rng_state; RNG continuity not restored")

    logger.info(
        "Resumed from %s at epoch %d step %d | LR %.2e",
        checkpoint_path,
        completed_epochs,
        global_step,
        scheduler.get_last_lr()[0],
    )
    return completed_epochs, global_step, best_val_loss, wandb_run_id


def advance_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler, steps: int
) -> None:
    for _ in range(steps):
        scheduler.step()


@dataclass(frozen=True)
class _WandbRun:
    id: str
    name: str


def init_wandb(
    cfg: TrainConfig,
    *,
    device: torch.device,
    effective_batch_size: int,
    vocab_size: int,
    num_params_m: float,
    train_dataset_size: int,
    val_dataset_size: int,
    resume_run_id: str | None,
    model_kwargs: dict[str, int],
) -> Any:
    if not cfg.use_wandb or cfg.wandb_mode == "disabled":
        logger.info("wandb logging disabled")
        return None

    init_kwargs: dict[str, Any] = {
        "project": cfg.wandb_project,
        "entity": cfg.wandb_entity,
        "name": cfg.wandb_run_name,
        "mode": cfg.wandb_mode,
        "config": {
            "batch_size": cfg.batch_size,
            "local_batch_size": cfg.local_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "num_epochs": cfg.num_epochs,
            "warmup_epochs": cfg.warmup_epochs,
            "min_lr_ratio": cfg.min_lr_ratio,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "max_grad_norm": cfg.max_grad_norm,
            "seed": cfg.seed,
            "cfg_drop_prob": cfg.cfg_drop_prob,
            "early_stop_patience": cfg.early_stop_patience,
            "dropout": cfg.dropout,
            "label_smoothing": cfg.label_smoothing,
            "num_workers": cfg.num_workers,
            "max_seq_len": cfg.pack_seq_len,
            "resume_mode": cfg.resume_mode,
            "device": str(device),
            "vocab_size": vocab_size,
            "num_params_m": num_params_m,
            "train_dataset_size": train_dataset_size,
            "val_dataset_size": val_dataset_size,
            **model_kwargs,
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
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pad_token_id: int,
    epoch: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
    global_step: int,
    use_wandb: bool,
    cfg: TrainConfig,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    vocab_size = unwrap_model(model).vocab_size

    optimizer.zero_grad(set_to_none=True)
    for step, (inputs, targets, doc_ids) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        doc_ids = doc_ids.to(device, non_blocking=True)

        assert inputs.shape == targets.shape == doc_ids.shape, (
            f"Got inputs={inputs.shape}, targets={targets.shape}, "
            f"doc_ids={doc_ids.shape}"
        )

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(inputs, doc_ids=doc_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=pad_token_id,
                label_smoothing=cfg.label_smoothing,
            )

        # Scale so accumulated gradients match the mean over the effective batch.
        # The last window of an epoch may be shorter than the accumulation size.
        window_start = step - step % cfg.gradient_accumulation_steps
        window_size = min(cfg.gradient_accumulation_steps, num_batches - window_start)
        scaler.scale(loss / window_size).backward()

        # Step on every Nth batch, and always flush on the final batch so the
        # last (possibly partial) accumulation window is not dropped.
        is_accumulation_boundary = (step + 1) % cfg.gradient_accumulation_steps == 0
        grad_norm: float | None = None
        if is_accumulation_boundary or (step + 1) == num_batches:
            scaler.unscale_(optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=cfg.max_grad_norm,
                ).item()
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        global_step += 1

        if (step + 1) % cfg.log_every == 0:
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
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if grad_norm is not None:
                    metrics["train/grad_norm"] = grad_norm
                wandb.log(metrics, step=global_step)

    return total_loss / max(num_batches, 1), global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    vocab_size = unwrap_model(model).vocab_size

    with torch.no_grad():
        for inputs, targets, doc_ids in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            doc_ids = doc_ids.to(device, non_blocking=True)

            assert inputs.shape == targets.shape == doc_ids.shape, (
                f"Got inputs={inputs.shape}, targets={targets.shape}, "
                f"doc_ids={doc_ids.shape}"
            )

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits, _ = model(inputs, doc_ids=doc_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=pad_token_id,
                    reduction="sum",
                )

            valid_tokens = int(targets.ne(pad_token_id).sum().item())
            total_loss += float(loss.item())
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise ValueError("Validation set contains no non-padding target tokens")
    return total_loss / total_tokens


def main(cfg: TrainConfig | None = None) -> None:
    if cfg is None:
        cfg = TrainConfig()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    set_seed(cfg.seed)
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

    batch_size = cfg.batch_size if device.type == "cuda" else cfg.local_batch_size

    logger.info("Device: %s | AMP dtype: %s | seed=%d", device, amp_dtype, cfg.seed)
    logger.info(
        "Batch size %d x %d accumulation steps = effective batch %d",
        batch_size,
        cfg.gradient_accumulation_steps,
        batch_size * cfg.gradient_accumulation_steps,
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    train_dataset, val_dataset, vocab_size, pad_token_id = load_datasets(
        cfg,
        TRAIN_TOKENS_PATH,
        VAL_TOKENS_PATH,
        TOKENIZER_PATH,
    )
    logger.info(
        "Loaded train=%d | validation=%d | vocab_size=%d | pad_token_id=%d",
        len(train_dataset),
        len(val_dataset),
        vocab_size,
        pad_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(batches_per_epoch / cfg.gradient_accumulation_steps)

    model_config = ModelConfig(vocab_size=vocab_size, dropout=cfg.dropout)
    model_kwargs = model_config_dict(model_config)

    # Build model first; weights-only resume loads weights before a fresh optimizer.
    model: nn.Module = MusicTransformer.from_config(model_config).to(device)
    num_params_m = unwrap_model(model).get_num_params()
    logger.info("Model parameters: %.2fM", num_params_m)

    completed_epochs = 0
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    wandb_run_id: str | None = None

    if cfg.resume_mode == "weights":
        completed_epochs, global_step, best_val_loss, wandb_run_id = resume_weights_only(
            cfg.resume_checkpoint_path,
            model,
            device,
            batches_per_epoch,
            cfg,
        )
    elif cfg.resume_mode == "full":
        logger.info("resume_mode='full'; full state resume including optimizer")

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg.weight_decay),
        lr=cfg.learning_rate,
        fused=device.type == "cuda",
    )
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    if cfg.resume_mode == "weights":
        advance_scheduler(scheduler, completed_epochs * steps_per_epoch)
        logger.info(
            "Fresh AdamW (wd=%.2f) and scheduler aligned to epoch %d | LR %.2e",
            cfg.weight_decay,
            completed_epochs,
            scheduler.get_last_lr()[0],
        )
    elif cfg.resume_mode == "full":
        latest_checkpoint = find_latest_checkpoint(CHECKPOINTS_DIR)
        completed_epochs, global_step, best_val_loss, wandb_run_id = resume_from_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            batches_per_epoch,
            steps_per_epoch,
        )
    else:
        logger.info("resume_mode='none', training from randomly initialized weights")

    wandb_run = init_wandb(
        cfg,
        device=device,
        effective_batch_size=batch_size * cfg.gradient_accumulation_steps,
        vocab_size=vocab_size,
        num_params_m=num_params_m,
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
        resume_run_id=wandb_run_id if cfg.resume_mode != "none" else None,
        model_kwargs=model_kwargs,
    )
    use_wandb = wandb_run is not None
    if use_wandb:
        wandb_run_id = wandb_run.id

    if device.type == "cuda":
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    start_epoch = completed_epochs + 1

    try:
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            avg_loss, global_step = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                pad_token_id,
                epoch,
                amp_dtype,
                use_amp,
                global_step,
                use_wandb,
                cfg,
            )
            val_loss = validate(
                model,
                val_loader,
                device,
                pad_token_id,
                amp_dtype,
                use_amp,
            )
            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                "Epoch %d finished | Train loss %.4f | Val loss %.4f | LR %.2e",
                epoch,
                avg_loss,
                val_loss,
                current_lr,
            )
            if use_wandb:
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "val_loss": val_loss,
                        "train/lr": current_lr,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            checkpoint_path = CHECKPOINTS_DIR / f"model_epoch_{epoch}.pt"
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=model_config,
                seed=cfg.seed,
                pad_token_id=pad_token_id,
                best_val_loss=min(best_val_loss, val_loss),
                wandb_run_id=wandb_run_id,
            )
            logger.info("Saved checkpoint to %s", checkpoint_path)
            prune_epoch_checkpoints(CHECKPOINTS_DIR, cfg.keep_last_checkpoints)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_checkpoint_path = CHECKPOINTS_DIR / "model_best.pt"
                save_checkpoint(
                    best_checkpoint_path,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    model_config=model_config,
                    seed=cfg.seed,
                    pad_token_id=pad_token_id,
                    best_val_loss=best_val_loss,
                    wandb_run_id=wandb_run_id,
                )
                logger.info(
                    "Saved new best checkpoint to %s | Val loss %.4f",
                    best_checkpoint_path,
                    best_val_loss,
                )
            else:
                epochs_without_improvement += 1
                logger.info(
                    "No val improvement for %d/%d epochs (best=%.4f)",
                    epochs_without_improvement,
                    cfg.early_stop_patience,
                    best_val_loss,
                )
                if epochs_without_improvement >= cfg.early_stop_patience:
                    logger.info(
                        "Early stopping at epoch %d | best val %.4f",
                        epoch,
                        best_val_loss,
                    )
                    break

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
