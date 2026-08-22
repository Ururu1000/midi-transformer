import torch
import torch.nn.functional as F

from musiclm.inference.sampler import apply_repetition_penalty, min_p_filter


class TestMinPFilter:
    def test_argmax_always_survives(self):
        logits = torch.tensor([10.0, 3.0, 1.0])
        filtered = min_p_filter(logits, min_p=0.99)
        assert filtered[0].item() != float("-inf")
        assert filtered.argmax().item() == 0

    def test_tail_suppressed(self):
        logits = torch.tensor([10.0, 2.0, 2.0, 2.0])
        filtered = min_p_filter(logits, min_p=0.5)
        # p(token) for the tail is ~e^-8 relative to top -> below 0.5 threshold.
        assert (filtered[1:] == float("-inf")).all()

    def test_zero_is_identity(self):
        logits = torch.tensor([1.0, -2.0, 0.3])
        out = min_p_filter(logits, min_p=0.0)
        assert torch.equal(out, logits)

    def test_output_is_valid_distribution_after_softmax(self):
        logits = torch.randn(200)
        filtered = min_p_filter(logits, min_p=0.05)
        probs = F.softmax(filtered, dim=-1)
        assert probs.sum().item() > 0.999


class TestRepetitionPenalty:
    def make(self):
        logits = torch.tensor([4.0, -4.0, 2.0, 1.0])
        window = torch.tensor([0, 1, 2])  # recently used ids 0,1,2
        is_pitch = torch.tensor([True, True, False, True])
        return logits, window, is_pitch

    def test_scales_not_shifts(self):
        logits, window, is_pitch = self.make()
        penalty = 2.0
        out = apply_repetition_penalty(logits.clone(), window, penalty, is_pitch)
        # Positive pitch logit divided, negative multiplied.
        assert torch.isclose(out[0], torch.tensor(2.0))
        assert torch.isclose(out[1], torch.tensor(-8.0))
        # Unseen id untouched.
        assert torch.isclose(out[3], torch.tensor(1.0))

    def test_grid_tokens_never_penalized(self):
        logits, window, is_pitch = self.make()
        out = apply_repetition_penalty(logits.clone(), window, 2.0, is_pitch)
        # id 2 was seen but is not a pitch token -> untouched.
        assert torch.isclose(out[2], torch.tensor(2.0))

    def test_penalty_one_is_identity(self):
        logits, window, is_pitch = self.make()
        out = apply_repetition_penalty(logits, window, 1.0, is_pitch)
        assert torch.equal(out, logits)

    def test_empty_window_is_identity(self):
        logits, _, is_pitch = self.make()
        empty = torch.tensor([], dtype=torch.long)
        out = apply_repetition_penalty(logits, empty, 1.5, is_pitch)
        assert torch.equal(out, logits)
