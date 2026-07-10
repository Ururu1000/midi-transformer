from __future__ import annotations

import logging

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
    # Relative position encoding via rotation of Q/K, shared across all layers.
    # Music depends on relative distances between notes, hence RoPE over absolute PE.
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

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    assert x.ndim == 4, f"Got {x.shape}"
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def build_causal_padding_attn_mask(
    attention_mask: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Convert (batch, seq) 1/0 mask into additive SDPA mask (batch, 1, seq, seq).

    Combines a lower-triangular causal mask with key padding: pad positions
    cannot be attended to. Used with is_causal=False because this PyTorch
    build rejects explicit attn_mask together with is_causal=True.
    """
    assert attention_mask.ndim == 2, f"Got {attention_mask.shape}"
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device

    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    key_valid = attention_mask.bool()[:, None, :]
    allowed = causal[None, :, :] & key_valid

    attn_mask = torch.zeros(
        batch_size,
        1,
        seq_len,
        seq_len,
        device=device,
        dtype=dtype,
    )
    return attn_mask.masked_fill(~allowed[:, None, :, :], torch.finfo(dtype).min)


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

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Got {x.shape}"

        qkv = self.qkv_proj(x)
        assert qkv.shape == (batch_size, seq_len, 3 * self.d_model), f"Got {qkv.shape}"

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        assert attn.shape == (
            batch_size,
            self.nhead,
            seq_len,
            self.head_dim,
        ), f"Got {attn.shape}"

        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(attn))
        assert out.shape == (batch_size, seq_len, self.d_model), f"Got {out.shape}"
        return out


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
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout, rotary)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 16,
        d_ff: int = 3072,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        rotary = RotaryEmbedding(d_model // nhead, max_seq_len)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, nhead, d_ff, dropout, rotary)
            for _ in range(num_layers)
        )
        self.norm_final = nn.LayerNorm(d_model)
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
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
        )

        x = self.embedding_dropout(self.token_embedding(input_ids))
        assert x.shape == (batch_size, seq_len, self.d_model), f"Got {x.shape}"

        attn_mask: Tensor | None = None
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len), (
                f"Got attention_mask={attention_mask.shape}, expected={(batch_size, seq_len)}"
            )
            # Full-length batches keep the fast is_causal SDPA path.
            if not bool(attention_mask.all()):
                attn_mask = build_causal_padding_attn_mask(attention_mask, x.dtype)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        assert logits.shape == (
            batch_size,
            seq_len,
            self.vocab_size,
        ), f"Got {logits.shape}"
        return logits

    def get_num_params(self) -> float:
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_params / 1e6


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = get_device()

    vocab_size = 500
    seq_len = 2048
    model = MusicTransformer(vocab_size=vocab_size).to(device)

    logger.info("Device: %s", device)
    logger.info("Trainable parameters: %.2fM", model.get_num_params())

    fake_input = torch.randint(0, vocab_size, (2, seq_len), device=device)
    logits = model(fake_input)

    logger.info("Input shape: %s", tuple(fake_input.shape))
    logger.info("Output shape: %s", tuple(logits.shape))
    assert logits.shape == (2, seq_len, vocab_size), f"Got {logits.shape}"
