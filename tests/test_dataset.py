import torch

from musiclm.data.dataset import PackedMusicDataset

PAD = 99


def make_doc(composer_id: int, n_tokens: int) -> dict:
    return {
        "composer_id": composer_id,
        "variants": {0: torch.arange(n_tokens, dtype=torch.int32) % 90},
    }


def build_dataset(docs, pack_seq_len=32, cfg_drop_prob=0.0):
    return PackedMusicDataset(
        docs=docs,
        pack_seq_len=pack_seq_len,
        pad_token_id=PAD,
        uncond_id=98,
        cfg_drop_prob=cfg_drop_prob,
        shifts=[0],
        augment=False,
    )


class TestPacking:
    def test_two_docs_share_one_row(self):
        docs = [make_doc(1, 8), make_doc(2, 6)]
        ds = build_dataset(docs)
        assert len(ds) == 1
        inputs, _targets, doc_ids = ds[0]

        # Row layout: [prefix, d0(8), prefix, d1(6), pad...]
        # indices:     0      1..8    9      10..15  16+
        assert inputs[0].item() == 1   # first doc's composer prefix
        assert inputs[9].item() == 2   # second doc's prefix right after doc0
        assert inputs[15].item() != PAD  # last content token of doc 1
        assert inputs[16].item() == PAD  # padding starts at 16

        assert doc_ids[0].item() == 0
        assert doc_ids[8].item() == 0
        assert doc_ids[9].item() == 1
        assert doc_ids[15].item() == 1
        assert doc_ids[16].item() == -1  # sentinel for padding

    def test_boundary_target_masked(self):
        docs = [make_doc(1, 8), make_doc(2, 6)]
        ds = build_dataset(docs)
        _, targets, _ = ds[0]
        # Position predicting the second doc's prefix must be ignored in loss.
        assert targets[8].item() == PAD

    def test_inputs_targets_are_shifted(self):
        docs = [make_doc(1, 10)]
        ds = build_dataset(docs)
        inputs, targets, _ = ds[0]
        # targets[t] must equal inputs[t+1] wherever the target is scored
        # (non-pad). Boundary/pad targets are excluded by their PAD value.
        next_inputs = torch.cat([inputs[1:], torch.full_like(inputs[:1], PAD)])
        valid = targets != PAD
        assert torch.equal(targets[valid], next_inputs[valid])

    def test_oversized_doc_truncated_to_row(self):
        docs = [make_doc(1, 100)]
        ds = build_dataset(docs, pack_seq_len=16)
        # slot = min(100+1, 16) = 16 -> one row fully occupied by the doc.
        assert len(ds) == 1
        inputs, _, doc_ids = ds[0]
        assert inputs.shape == (15,)
        # Content fills the whole row; nothing is padding.
        assert (doc_ids[:15] == 0).all()
        assert (inputs != PAD).all()


class TestConditioning:
    def test_cfg_drop_swaps_prefix(self):
        docs = [make_doc(7, 4)]
        ds = build_dataset(docs, cfg_drop_prob=1.0)  # always drop
        inputs, _, _ = ds[0]
        assert inputs[0].item() == 98  # unconditional token replaces composer id

    def test_no_drop_keeps_composer(self):
        docs = [make_doc(7, 4)]
        ds = build_dataset(docs, cfg_drop_prob=0.0)
        inputs, _, _ = ds[0]
        assert inputs[0].item() == 7
