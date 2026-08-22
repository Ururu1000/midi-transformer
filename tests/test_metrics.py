import numpy as np
import pytest

from musiclm.evaluation.metrics import (
    _histogram_kld,
    detect_key,
    scale_consistency,
)


def histogram_for(pitch_classes: dict[int, float]) -> np.ndarray:
    hist = np.zeros(12)
    for pc, weight in pitch_classes.items():
        hist[pc % 12] += weight
    return hist


class TestDetectKey:
    def test_c_major(self):
        pch = histogram_for({0: 6.3, 4: 3.5, 7: 4.0, 2: 2.2, 5: 2.0, 9: 2.0, 11: 1.0})
        root, mode = detect_key(pch)
        assert (root, mode) == ("C", "major")

    def test_a_minor(self):
        pch = histogram_for({9: 6.3, 0: 3.5, 4: 3.5, 2: 2.7, 5: 2.6, 7: 2.5, 10: 2.0})
        root, mode = detect_key(pch)
        assert (root, mode) == ("A", "minor")


class TestScaleConsistency:
    def test_diatonic_notes_score_one(self):
        assert scale_consistency([60, 62, 64, 67], "C", "major") == 1.0

    def test_chromatic_outliers_lower_score(self):
        score = scale_consistency([60, 61, 64, 66], "C", "major")
        assert 0.0 < score < 1.0

    def test_empty_is_zero(self):
        assert scale_consistency([], "C", "major") == 0.0


class TestHistogramKLD:
    def test_identical_samples_zero(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(size=500)
        kld = _histogram_kld(samples, samples, bins=20, label="test")
        assert abs(kld) < 1e-6

    def test_disjoint_shift_positive(self):
        rng = np.random.default_rng(0)
        a = rng.normal(loc=0.0, size=500)
        b = rng.normal(loc=50.0, size=500)
        kld = _histogram_kld(a, b, bins=20, label="test")
        assert kld > 1.0

    def test_empty_returns_nan(self):
        import math

        empty = np.array([])
        assert math.isnan(_histogram_kld(empty, np.array([1.0]), 10, "t"))


class TestFileMetricsDataclass:
    def test_defaults_allow_construction(self):
        from pathlib import Path

        from musiclm.evaluation.metrics import FileMetrics

        m = FileMetrics(path=Path("x.mid"), detected_key="C major",
                        scale_consistency=1.0, pc_entropy=2.0, pitch_range=12,
                        polyphony_rate=1.5, note_density=3.0,
                        ioi_mean=0.25, ioi_std=0.05)
        assert m.note_count == 0 and m.compression_ratio == 1.0


@pytest.mark.parametrize("bins", [5, 30])
def test_kld_bins_do_not_crash(bins):
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=100), rng.normal(size=100)
    assert isinstance(_histogram_kld(a, b, bins=bins, label="t"), float)
