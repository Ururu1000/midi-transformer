from __future__ import annotations

import logging
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self, head_dim: int, max_seq_len: int, base: float = 10000.0
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0, f"Got head_dim={head_dim}"

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, start_pos: int = 0) -> tuple[Tensor, Tensor]:
        end_pos = start_pos + seq_len
        return self.cos_cached[start_pos:end_pos], self.sin_cached[start_pos:end_pos]


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    assert x.ndim == 4, f"Got {x.shape}"
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(d_ff * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class KVCache(NamedTuple):
    key: Tensor
    value: Tensor


def build_document_causal_mask(doc_ids: Tensor) -> Tensor:
    """Block-diagonal causal mask for packed sequences.

    ``doc_ids`` holds one document index per position; padding uses its own
    sentinel id. A position may only attend to earlier positions of the same
    document, so packed rows never leak attention across document boundaries.
    """
    assert doc_ids.ndim == 2, f"Got {doc_ids.shape}"
    batch_size, seq_len = doc_ids.shape
    same_document = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(2)
    causal = torch.ones(
        seq_len, seq_len, dtype=torch.bool, device=doc_ids.device
    ).tril()
    mask = (same_document & causal).unsqueeze(1)
    assert mask.shape == (batch_size, 1, seq_len, seq_len), f"Got {mask.shape}"
    return mask


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        rotary: RotaryEmbedding,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"Got d_model={d_model}, nhead={nhead}"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.rotary = rotary

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        past_key_value: KVCache | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, KVCache | None]:
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Got {x.shape}"

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len, start_pos=start_pos)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value.key, k], dim=2)
            v = torch.cat([past_key_value.value, v], dim=2)

        next_cache = KVCache(key=k, value=v) if use_cache else None

        query_len = q.shape[2]
        key_len = k.shape[2]
        is_causal = attn_mask is None and query_len == key_len
        if attn_mask is not None:
            # Precomputed block-diagonal causal mask for packed training rows;
            # incompatible with an incremental KV cache by construction.
            assert past_key_value is None and not use_cache
            assert attn_mask.shape == (batch_size, 1, query_len, key_len), (
                f"Got {attn_mask.shape}"
            )
        elif not is_causal:
            # is_causal aligns the mask to the top-left corner, which is only
            # correct when queries cover the whole key sequence. With a cache the
            # queries are the tail of the keys, so the mask is offset by the
            # cached length.
            attn_mask = torch.ones(
                query_len, key_len, dtype=torch.bool, device=q.device
            ).tril(diagonal=key_len - query_len)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(attn))
        return out, next_cache


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        dropout: float,
        rotary: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout, rotary)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout)

    def forward(
        self,
        x: Tensor,
        past_key_value: KVCache | None = None,
        start_pos: int = 0,
        use_cache: bool = False,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, KVCache | None]:
        normed = self.norm1(x)
        attn_out, next_cache = self.attn(
            normed,
            past_key_value=past_key_value,
            start_pos=start_pos,
            use_cache=use_cache,
            attn_mask=attn_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, next_cache


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 24,
        d_ff: int = 3072,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        rotary = RotaryEmbedding(d_model // nhead, max_seq_len)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, nhead, d_ff, dropout, rotary)
            for _ in range(num_layers)
        )
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
        doc_ids: Tensor | None = None,
    ) -> tuple[Tensor, list[KVCache | None] | None]:
        batch_size, seq_len = input_ids.shape
        attn_mask: Tensor | None = None
        if doc_ids is not None:
            assert past_key_values is None and not use_cache, (
                "doc_ids masking is a training-only path"
            )
            assert doc_ids.shape == input_ids.shape, f"Got {doc_ids.shape}"
            attn_mask = build_document_causal_mask(doc_ids)

        start_pos = 0
        if past_key_values is not None:
            assert len(past_key_values) == self.num_layers, (
                f"Expected {self.num_layers} cached layers, got {len(past_key_values)}"
            )
            first_cache = past_key_values[0]
            if first_cache is not None:
                start_pos = first_cache.key.shape[2]

        assert start_pos + seq_len <= self.max_seq_len, (
            f"start_pos={start_pos} + seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
        )

        x = self.embedding_dropout(self.token_embedding(input_ids))
        assert x.shape == (batch_size, seq_len, self.d_model), f"Got {x.shape}"

        next_caches: list[KVCache | None] = [] if use_cache else []
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = past_key_values[layer_idx] if past_key_values is not None else None
            x, cache = block(
                x,
                past_key_value=layer_cache,
                start_pos=start_pos,
                use_cache=use_cache,
                attn_mask=attn_mask,
            )
            if use_cache:
                next_caches.append(cache)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        assert logits.shape == (
            batch_size,
            seq_len,
            self.vocab_size,
        ), f"Got {logits.shape}"

        return logits, (next_caches if use_cache else None)

    def get_num_params(self) -> float:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_params / 1e6


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = get_device()

    vocab_size = 500
    seq_len = 128
    model = MusicTransformer(vocab_size=vocab_size).to(device)

    logger.info("Device: %s", device)
    logger.info("Trainable parameters: %.2fM", model.get_num_params())

    fake_input = torch.randint(0, vocab_size, (2, seq_len), device=device)
    logits, _ = model(fake_input)

    logger.info("Input shape: %s", tuple(fake_input.shape))
    logger.info("Output shape: %s", tuple(logits.shape))
    assert logits.shape == (2, seq_len, vocab_size), f"Got {logits.shape}"

    doc_ids = torch.zeros((2, seq_len), dtype=torch.long, device=device)
    doc_ids[:, seq_len // 2 :] = 1
    packed_logits, _ = model(fake_input, doc_ids=doc_ids)
    assert packed_logits.shape == (2, seq_len, vocab_size), f"Got {packed_logits.shape}"
    logger.info("Packed forward with block-diagonal mask OK")
