import math

import pytest
import torch

from musiclm.checkpoint import export_inference_weights
from musiclm.config import ModelConfig, TrainConfig
from musiclm.data.dataset import PackedMusicDataset
from musiclm.inference.sampler import load_model
from musiclm.model import MusicTransformer
from musiclm.training.trainer import (
    WARMUP_START_FACTOR,
    build_param_groups,
    build_scheduler,
    lr_multiplier,
    prune_epoch_checkpoints,
    resume_weights_only,
    save_checkpoint,
    train_one_epoch,
    validate,
)

PAD = 0
VOCAB = 40
# Deliberately not the default architecture: loading must read it back.
TINY = ModelConfig(
    vocab_size=VOCAB, d_model=32, nhead=4, num_layers=2, d_ff=48, max_seq_len=64,
    dropout=0.0,
)


def tiny_model() -> MusicTransformer:
    torch.manual_seed(0)
    return MusicTransformer.from_config(TINY)


def tiny_loader(num_docs: int, batch_size: int = 1) -> torch.utils.data.DataLoader:
    generator = torch.Generator().manual_seed(0)
    docs = [
        {
            "composer_id": 1,
            "variants": {0: torch.randint(3, VOCAB, (20,), generator=generator).int()},
        }
        for _ in range(num_docs)
    ]
    dataset = PackedMusicDataset(
        docs, pack_seq_len=32, pad_token_id=PAD, uncond_id=2, cfg_drop_prob=0.0,
        shifts=[0], augment=False,
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def tiny_training_state(cfg: TrainConfig, steps_per_epoch: int):
    model = tiny_model()
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg.weight_decay), lr=1e-3)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, scheduler, scaler


class TestParamGroups:
    def test_only_matrices_decay(self):
        model = tiny_model()
        decay, no_decay = build_param_groups(model, weight_decay=0.1)
        assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
        assert all(p.ndim >= 2 for p in decay["params"])
        assert all(p.ndim < 2 for p in no_decay["params"])
        # The tied embedding/lm_head tensor appears exactly once.
        grouped = len(decay["params"]) + len(no_decay["params"])
        assert grouped == len(list(model.parameters()))


class TestLrSchedule:
    def test_warmup_then_cosine_to_floor(self):
        warmup, total, floor = 10, 100, 0.1
        assert lr_multiplier(0, warmup, total, floor) == pytest.approx(WARMUP_START_FACTOR)
        assert lr_multiplier(warmup, warmup, total, floor) == pytest.approx(1.0)
        assert lr_multiplier(total, warmup, total, floor) == pytest.approx(floor)
        assert lr_multiplier(total + 50, warmup, total, floor) == pytest.approx(floor)
        decay = [lr_multiplier(s, warmup, total, floor) for s in range(warmup, total)]
        assert decay == sorted(decay, reverse=True)

    def test_warmup_rises_every_step(self):
        values = [lr_multiplier(s, 10, 100, 0.1) for s in range(11)]
        assert values == sorted(values) and len(set(values)) == len(values)


class TestTrainOneEpoch:
    def test_partial_window_steps_and_learns(self):
        # 5 rows of one batch each with accumulation 2 -> windows of 2, 2, 1.
        loader = tiny_loader(num_docs=5)
        assert len(loader) == 5
        cfg = TrainConfig(gradient_accumulation_steps=2, log_every=1000, warmup_epochs=0)
        steps_per_epoch = math.ceil(len(loader) / cfg.gradient_accumulation_steps)
        model, optimizer, scheduler, scaler = tiny_training_state(cfg, steps_per_epoch)
        before = validate(model, loader, torch.device("cpu"), PAD, torch.float32, False)

        for epoch in range(1, 4):
            loss, global_step = train_one_epoch(
                model, loader, optimizer, scheduler, scaler, torch.device("cpu"), PAD,
                epoch, torch.float32, False, (epoch - 1) * len(loader), False, cfg,
            )
            assert math.isfinite(loss)
        assert global_step == 3 * len(loader)
        assert scheduler.last_epoch == 3 * steps_per_epoch

        after = validate(model, loader, torch.device("cpu"), PAD, torch.float32, False)
        assert after < before


class TestCheckpoints:
    def save(self, path, model):
        cfg = TrainConfig(seed=7)
        _, optimizer, scheduler, scaler = tiny_training_state(cfg, steps_per_epoch=1)
        save_checkpoint(
            path, epoch=1, global_step=3, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, model_config=TINY, seed=cfg.seed,
            pad_token_id=PAD, best_val_loss=1.5,
        )

    def test_load_model_reads_architecture(self, tmp_path):
        model = tiny_model()
        path = tmp_path / "model_best.pt"
        self.save(path, model)

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["seed"] == 7
        assert checkpoint["model_config"]["d_model"] == TINY.d_model

        loaded = load_model(path, torch.device("cpu"))
        assert loaded.num_layers == TINY.num_layers
        for key, value in model.state_dict().items():
            assert torch.equal(loaded.state_dict()[key], value)

    def test_safetensors_export_round_trip(self, tmp_path):
        model = tiny_model().eval()
        path = tmp_path / "model_best.pt"
        self.save(path, model)

        weights = export_inference_weights(path, tmp_path / "export", torch.float32)
        loaded = load_model(weights, torch.device("cpu"))
        tokens = torch.randint(0, VOCAB, (1, 16))
        with torch.no_grad():
            assert torch.allclose(loaded(tokens)[0], model(tokens)[0])
        # Tie between embedding and output head survives the export.
        assert loaded.token_embedding.weight is loaded.lm_head.weight

    def test_prune_keeps_newest_and_best(self, tmp_path):
        for epoch in range(1, 5):
            (tmp_path / f"model_epoch_{epoch}.pt").touch()
        (tmp_path / "model_best.pt").touch()
        prune_epoch_checkpoints(tmp_path, keep=2)
        remaining = sorted(path.name for path in tmp_path.iterdir())
        assert remaining == ["model_best.pt", "model_epoch_3.pt", "model_epoch_4.pt"]

    def test_prune_disabled(self, tmp_path):
        for epoch in range(1, 4):
            (tmp_path / f"model_epoch_{epoch}.pt").touch()
        prune_epoch_checkpoints(tmp_path, keep=0)
        assert len(list(tmp_path.iterdir())) == 3

    def test_weights_resume_without_checkpoint_starts_fresh(self, tmp_path):
        counters = resume_weights_only(
            tmp_path / "missing.pt", tiny_model(), torch.device("cpu"), 10, TrainConfig()
        )
        assert counters == (0, 0, float("inf"), None)
