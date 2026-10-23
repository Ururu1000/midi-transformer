from pathlib import Path

import pytest

from musiclm.config import GenerateConfig, TrainConfig, with_overrides
from musiclm.inference import cli as generate_cli
from musiclm.inference import prompts
from musiclm.inference.prompts import LONG_LENGTH
from musiclm.training import cli as train_cli


@pytest.fixture
def captured_train_cfg(monkeypatch):
    captured = []
    monkeypatch.setattr(train_cli, "run_training", captured.append)
    return captured


@pytest.fixture
def captured_generate_cfg(monkeypatch):
    captured = []

    def fake_generation(cfg):
        captured.append(cfg)
        return Path(cfg.output_path)

    monkeypatch.setattr(generate_cli, "run_generation", fake_generation)
    return captured


class TestWithOverrides:
    def test_none_keeps_default(self):
        cfg = with_overrides(TrainConfig(), num_epochs=None, learning_rate=1e-3)
        assert cfg.num_epochs == TrainConfig().num_epochs
        assert cfg.learning_rate == 1e-3

    def test_falsy_values_still_override(self):
        cfg = with_overrides(TrainConfig(), use_wandb=False, seed=0)
        assert cfg.use_wandb is False
        assert cfg.seed == 0


class TestTrainCli:
    def test_no_flags_gives_defaults(self, captured_train_cfg):
        train_cli.main([])
        assert captured_train_cfg == [TrainConfig()]

    def test_flags_override(self, captured_train_cfg, tmp_path):
        resume = tmp_path / "ckpt.pt"
        train_cli.main([
            "--resume-mode", "none", "--epochs", "3", "--lr", "1e-4",
            "--accum", "2", "--no-wandb", "--resume-checkpoint", str(resume),
        ])
        (cfg,) = captured_train_cfg
        assert cfg.resume_mode == "none"
        assert cfg.num_epochs == 3
        assert cfg.learning_rate == 1e-4
        assert cfg.gradient_accumulation_steps == 2
        assert cfg.use_wandb is False
        assert cfg.resume_checkpoint_path == resume


class TestGenerateCli:
    def test_no_flags_gives_defaults(self, captured_generate_cfg):
        generate_cli.main([])
        assert captured_generate_cfg == [GenerateConfig()]

    def test_flags_override(self, captured_generate_cfg):
        generate_cli.main([
            "--composer", "Johann Sebastian Bach", "--length", "300",
            "--candidates", "2", "--seed", "5", "--keep-primer",
        ])
        (cfg,) = captured_generate_cfg
        assert cfg.composer == "Johann Sebastian Bach"
        assert cfg.length == 300
        assert cfg.num_candidates == 2
        assert cfg.seed == 5
        assert cfg.keep_primer is True

    def test_prompt_fills_unset_fields_and_flags_win(self, captured_generate_cfg, monkeypatch):
        monkeypatch.setattr(
            prompts,
            "load_available_composers",
            lambda **_: ["Frédéric Chopin", "Johann Sebastian Bach"],
        )
        generate_cli.main(["--prompt", "generate bach, longer", "--temperature", "0.5"])
        (cfg,) = captured_generate_cfg
        assert cfg.composer == "Johann Sebastian Bach"
        assert cfg.length == LONG_LENGTH
        assert cfg.temperature == 0.5
