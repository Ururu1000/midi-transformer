from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"Got d_model={d_model}, nhead={nhead}"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Got {x.shape}"

        qkv = self.qkv_proj(x)
        assert qkv.shape == (batch_size, seq_len, 3 * self.d_model), f"Got {qkv.shape}"

        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
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
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)
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

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}"
        )

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)
        assert x.shape == (batch_size, seq_len, self.d_model), f"Got {x.shape}"

        for block in self.blocks:
            x = block(x)

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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
