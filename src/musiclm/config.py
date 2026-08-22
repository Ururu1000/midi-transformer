"""Single source of truth for paths and hyperparameter configurations.

All defaults mirror the values the shipped checkpoint was trained with; CLI
flags override them per run. Paths are anchored at the repository root so
entry points work from any working directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# src/musiclm/config.py -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi"
MAESTRO_METADATA_PATH = RAW_MIDI_DIR / "maestro-v3.0.0.csv"
GIANTMIDI_DIR = RAW_MIDI_DIR / "giantmidi"

PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZER_PATH = PROCESSED_DIR / "tokenizer.json"
COMPOSER_MAPPING_PATH = PROCESSED_DIR / "composer_mapping.json"
TRAIN_TOKENS_PATH = PROCESSED_DIR / "tokens_train.pt"
VAL_TOKENS_PATH = PROCESSED_DIR / "tokens_val.pt"
UNIFIED_METADATA_PATH = PROCESSED_DIR / "unified_metadata.csv"

GENERATED_DIR = DATA_DIR / "generated"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT_PATH = CHECKPOINTS_DIR / "model_best.pt"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters. Defaults match the released checkpoints."""

    vocab_size: int
    d_model: int = 768
    nhead: int = 12
    num_layers: int = 24
    d_ff: int = 3072
    max_seq_len: int = 4096
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    # L4 24GB: seq 4096 attention is ~4x seq 2048. Batch 8 + accum 16 gives an
    # effective batch of 128. The block-diagonal attn_mask routes SDPA to the
    # memory-efficient backend (flash kernels reject arbitrary masks); drop
    # further if the GPU OOMs.
    batch_size: int = 8
    # MPS/CPU have no flash-attention kernel, so SDPA materializes the full
    # (batch, heads, seq, seq) score matrix. Keep local smoke tests tiny.
    local_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: int = 50
    warmup_epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.02
    max_grad_norm: float = 1.0
    log_every: int = 10
    seed: int = 42
    # Stop once validation fails to improve for this many epochs in a row.
    early_stop_patience: int = 5
    dropout: float = 0.1
    label_smoothing: float = 0.0
    # Fraction of training rows whose composer tokens are swapped for the
    # unconditional token, so the same weights model p(x) and p(x | composer).
    cfg_drop_prob: float = 0.15
    # DataLoader workers for the Linux VM; pinned memory speeds up host->GPU.
    num_workers: int = 4
    # Whole documents are bin-packed into pack_seq_len rows. Every row carries
    # doc_ids so attention is block-diagonal: no token ever attends across a
    # document boundary or into padding.
    pack_seq_len: int = 4096

    # Resume policy: "none" trains from scratch, "weights" loads model weights
    # plus epoch/step counters but keeps optimizer/scheduler/scaler fresh,
    # "full" restores every training state from the latest epoch checkpoint.
    resume_mode: str = "weights"  # "none" | "weights" | "full"
    resume_checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH

    use_wandb: bool = True
    wandb_project: str = "ai-music-project"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    # "online" | "offline" | "disabled" — offline is useful on air-gapped VMs.
    wandb_mode: str = "online"


@dataclass(frozen=True)
class GenerateConfig:
    checkpoint_path: Path = CHECKPOINTS_DIR / "model_best_ancient_tree17.pt"
    tokenizer_path: Path = TOKENIZER_PATH
    output_path: Path = PROCESSED_DIR / "output.mid"

    composer: str = "Frédéric Chopin"
    length: int = 1024
    temperature: float = 0.92
    # Min-P keeps every token whose probability is at least min_p times the top
    # token's probability. 0.02 allows subtle chord tones and polyphonic
    # textures to survive.
    min_p: float = 0.02
    # Softened CFG (1.2 vs 1.5) restores composer steering without suppressing
    # secondary voices and outer registers.
    cfg_scale: float = 1.2
    # Classical motifs reuse pitches; default off. Pitch-only window still
    # available.
    repetition_penalty: float = 1.0
    # Penalize against recent context only. A whole-sequence window makes every
    # pitch progressively unusable, which starves long generations of material.
    penalty_window: int = 64
    # The training chunks never contained EOS, so its logit is unreliable early
    # on.
    min_new_tokens: int = 96
    # KV cache is numerically identical to full-prefix decoding; keep it on for
    # speed.
    use_kv_cache: bool = True
    seed: int | None = None


@dataclass(frozen=True)
class EvalConfig:
    csv_output_path: Path | None = None


FORBIDDEN_PREFIXES = ("PAD", "BOS", "MASK", "Composer")
PITCH_PREFIXES = ("Pitch", "PitchDrum")
# Tokens that carry the rhythmic grid. Penalizing these starves the model of
# the bar/beat vocabulary it must reuse constantly, which collapses timing.
GRID_PREFIXES = ("Bar", "Position", "TimeShift", "Rest", "Chord", "Tempo")


def model_config_dict(cfg: ModelConfig) -> dict[str, int]:
    """Flat dict of architecture fields, e.g. for experiment tracking."""
    return {
        "d_model": cfg.d_model,
        "nhead": cfg.nhead,
        "num_layers": cfg.num_layers,
        "d_ff": cfg.d_ff,
        "max_seq_len": cfg.max_seq_len,
    }
