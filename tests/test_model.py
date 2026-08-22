import torch

from musiclm.model import (
    KVCache,
    MusicTransformer,
    build_document_causal_mask,
)


def make_tiny_model(vocab_size=50, seed=0):
    torch.manual_seed(seed)
    return MusicTransformer(
        vocab_size=vocab_size,
        d_model=32,
        nhead=4,
        num_layers=2,
        d_ff=48,
        max_seq_len=64,
        dropout=0.0,
    ).eval()


class TestDocumentMask:
    def test_block_diagonal_causal(self):
        doc_ids = torch.tensor([[0, 0, 0, 1, 1]])
        mask = build_document_causal_mask(doc_ids)
        assert mask.shape == (1, 1, 5, 5)
        m = mask[0, 0]

        # Doc 0 positions attend to earlier doc-0 positions only.
        assert m[0, :1].all() and not m[0, 1:].any()
        assert m[1, :2].all() and not m[1, 2:].any()
        assert m[2, :3].all() and not m[2, 3:].any()

        # Doc 1 starts fresh: no attention into doc 0.
        assert m[3].tolist() == [False, False, False, True, False]
        assert m[4].tolist() == [False, False, False, True, True]

    def test_padding_sentinel_isolated(self):
        # PAD_DOC_ID = -1 must never collide with real segment ordinals:
        # real tokens must never attend into padding.
        doc_ids = torch.tensor([[0, 0, -1, -1]])
        mask = build_document_causal_mask(doc_ids)
        m = mask[0, 0]
        # Real doc-0 tokens never see the pad tail.
        assert not m[0, 2:].any()
        assert not m[1, 2:].any()
        # Pad positions only attend within the pad region (harmless: their
        # losses are ignored via ignore_index).
        assert not m[2, :2].any()
        assert m[2, 2] and m[3, 2]


class TestForwardEquivalence:
    def test_kv_cache_matches_full_forward(self):
        model = make_tiny_model()
        seq_len = 12
        tokens = torch.randint(0, 50, (1, seq_len))

        full_logits, _ = model(tokens)

        past = None
        for step in range(seq_len):
            context = tokens[:, : step + 1] if past is None else tokens[:, step : step + 1]
            logits, past = model(context, past_key_values=past, use_cache=True)
            last = logits[:, -1, :]
            if step == 0:
                collected = [last]
            else:
                collected.append(last)

        incremental = torch.cat(collected, dim=0)
        assert torch.allclose(incremental, full_logits[0], atol=1e-5)

    def test_cache_grows_and_types(self):
        model = make_tiny_model()
        tokens = torch.randint(0, 50, (1, 4))
        _, past = model(tokens, use_cache=True)
        assert isinstance(past, list) and len(past) == 2
        assert isinstance(past[0], KVCache)
        assert past[0].key.shape == (1, 4, 4, 8)  # batch, seq, heads, head_dim

        _, past2 = model(tokens[:, 3:], past_key_values=past, use_cache=True)
        assert past2[0].key.shape[2] == 5


class TestModelConfig:
    def test_from_config_same_keys(self):
        from musiclm.config import ModelConfig

        torch.manual_seed(0)
        cfg = ModelConfig(vocab_size=50, d_model=32, nhead=4, num_layers=2,
                          d_ff=48, max_seq_len=64, dropout=0.0)
        a = MusicTransformer(vocab_size=50, d_model=32, nhead=4, num_layers=2,
                             d_ff=48, max_seq_len=64, dropout=0.0)
        b = MusicTransformer.from_config(cfg)
        assert set(a.state_dict()) == set(b.state_dict())
