import pytest
from huggingface_hub.utils import EntryNotFoundError

from musiclm import hub


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Record Hub downloads instead of performing them."""
    calls = []

    def fake_download(filename, repo_id):
        calls.append(filename)
        return tmp_path / "hub" / filename

    monkeypatch.setattr(hub, "_download", fake_download)
    return calls


class TestResolveReleaseFile:
    def test_existing_and_explicit_paths_untouched(self, downloads, tmp_path):
        present = tmp_path / "tokenizer.json"
        present.touch()
        assert hub.resolve_release_file(present) == present
        explicit = tmp_path / "elsewhere.json"
        assert hub.resolve_release_file(explicit) == explicit
        assert downloads == []

    def test_missing_default_is_downloaded(self, downloads, monkeypatch, tmp_path):
        default = tmp_path / "processed" / "tokenizer.json"
        monkeypatch.setitem(hub.RELEASE_FILES, default, "tokenizer.json")
        assert hub.resolve_release_file(default) == tmp_path / "hub" / "tokenizer.json"
        assert downloads == ["tokenizer.json"]


class TestResolveCheckpoint:
    @pytest.fixture
    def default_checkpoint(self, monkeypatch, tmp_path):
        path = tmp_path / "checkpoints" / "model_best.pt"

        class Defaults:
            checkpoint_path = path

        monkeypatch.setattr(hub, "GenerateConfig", Defaults)
        return path

    def test_explicit_path_untouched(self, downloads, default_checkpoint, tmp_path):
        explicit = tmp_path / "mine.pt"
        assert hub.resolve_checkpoint(explicit) == explicit
        assert downloads == []

    def test_prefers_safetensors_export(self, downloads, default_checkpoint, tmp_path):
        resolved = hub.resolve_checkpoint(default_checkpoint)
        assert resolved == tmp_path / "hub" / hub.HUB_WEIGHTS
        assert downloads == [hub.HUB_WEIGHTS_CONFIG, hub.HUB_WEIGHTS]

    def test_falls_back_to_training_checkpoint(self, monkeypatch, default_checkpoint, tmp_path):
        calls = []

        def fake_download(filename, repo_id):
            calls.append(filename)
            if filename == hub.HUB_WEIGHTS_CONFIG:
                raise EntryNotFoundError("no export published")
            return tmp_path / filename

        monkeypatch.setattr(hub, "_download", fake_download)
        assert hub.resolve_checkpoint(default_checkpoint) == tmp_path / hub.HUB_TRAINING_CHECKPOINT
        assert calls == [hub.HUB_WEIGHTS_CONFIG, hub.HUB_TRAINING_CHECKPOINT]
