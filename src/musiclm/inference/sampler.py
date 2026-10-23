"""Autoregressive sampler: KV-cached decoding, CFG, min-p, repetition penalty."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from miditok import REMI, TokSequence
from torch import Tensor

from musiclm.checkpoint import load_weights
from musiclm.config import (
    FORBIDDEN_PREFIXES,
    GRID_PREFIXES,
    PITCH_PREFIXES,
    GenerateConfig,
)
from musiclm.data.tokenizer import (
    UNCONDITIONAL_COMPOSER,
    ComposerREMI,
    composer_vocab_token,
    iter_vocab_entries,
    learned_token_id,
    list_composer_tokens,
)
from musiclm.hub import resolve_checkpoint, resolve_release_file
from musiclm.inference.priming import drop_primer, load_primer
from musiclm.inference.rerank import load_reference_stats, score_candidates
from musiclm.model import KVCache, MusicTransformer, get_device

logger = logging.getLogger(__name__)


class TokenCategories:
    def __init__(self, tokenizer: REMI, device: torch.device) -> None:
        forbidden: list[int] = []
        pitch: list[int] = []
        vocab_size = len(tokenizer)
        for token_id, base_tokens in iter_vocab_entries(tokenizer):
            prefixes = {token.split("_")[0] for token in base_tokens}
            if prefixes & set(FORBIDDEN_PREFIXES):
                forbidden.append(token_id)
            # A BPE id may merge several base tokens. It only counts as a pitch
            # token when it carries a pitch and no grid token, so penalizing it
            # can never suppress a Bar/Position the model needs to keep time.
            if prefixes & set(PITCH_PREFIXES) and not prefixes & set(GRID_PREFIXES):
                pitch.append(token_id)

        self.eos_token_id = learned_token_id(tokenizer, "EOS_None")
        self.forbidden_ids = torch.tensor(
            sorted(set(forbidden) - {self.eos_token_id}),
            dtype=torch.long,
            device=device,
        )
        self.is_pitch = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        if pitch:
            self.is_pitch[torch.tensor(sorted(pitch), device=device)] = True
        logger.info(
            "Repetition penalty applies to %d/%d pitch-only ids",
            int(self.is_pitch.sum().item()),
            vocab_size,
        )


def load_model(checkpoint_path: Path, device: torch.device) -> MusicTransformer:
    """Load a training checkpoint (``.pt``) or an export (``.safetensors``).

    The default checkpoint is downloaded from the Hub when missing locally.
    """
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    state_dict, config = load_weights(checkpoint_path)
    model = MusicTransformer.from_config(config)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not match the current architecture "
            "(RMSNorm + SwiGLU + KV-cache). Checkpoints trained before this change "
            "must be regenerated with `musiclm-train`."
        ) from exc
    del state_dict
    model.to(device)
    model.eval()
    logger.info(
        "Loaded model from %s | vocab_size=%d | %.2fM params",
        checkpoint_path,
        config.vocab_size,
        model.get_num_params(),
    )
    return model


def load_tokenizer(tokenizer_path: Path) -> ComposerREMI:
    """Load the tokenizer; the default one is downloaded from the Hub if missing."""
    tokenizer_path = resolve_release_file(tokenizer_path)
    tokenizer = ComposerREMI(params=str(tokenizer_path))
    logger.info(
        "Loaded tokenizer from %s | vocab_size=%d | bpe=%s",
        tokenizer_path,
        len(tokenizer),
        tokenizer.is_trained,
    )
    return tokenizer


def resolve_composer_token(tokenizer: REMI, composer: str) -> str:
    token = composer_vocab_token(composer)
    if token in tokenizer.vocab:
        return token

    available = list_composer_tokens(tokenizer)
    raise KeyError(
        f"Composer token {token!r} missing from tokenizer. Available: {available}"
    )


def pick_start_token(tokenizer: REMI, composer: str) -> int:
    token = resolve_composer_token(tokenizer, composer)
    start_token = learned_token_id(tokenizer, token)
    logger.info("Using composer seed %s (id=%d)", token, start_token)
    return start_token


def pick_unconditional_token(tokenizer: REMI) -> int | None:
    token = composer_vocab_token(UNCONDITIONAL_COMPOSER)
    if token not in tokenizer.vocab:
        return None
    return learned_token_id(tokenizer, token)


def min_p_filter(logits: Tensor, min_p: float) -> Tensor:
    """Drop tokens below ``min_p`` x the top probability; works on 1-D or (N, V)."""
    assert logits.ndim in (1, 2), f"Got {logits.shape}"
    if min_p <= 0.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    # The argmax always survives, so the filtered distribution is never empty.
    return logits.masked_fill(probs < threshold, float("-inf"))


def apply_repetition_penalty(
    logits: Tensor,
    window: Tensor,
    penalty: float,
    is_penalizable: Tensor,
) -> Tensor:
    """Divide recently-used positive pitch logits, multiply negative ones.

    Scaling instead of subtracting keeps the penalty proportional to how much
    the model wants a token, so confident-but-repeated ids fall furthest.
    Structural, temporal and special ids are never touched: REMI reuses them on
    every note, so penalizing them would break the bar/beat grid.

    Accepts a single row (``logits`` (V,), ``window`` (W,)) or a batch
    (``logits`` (N, V), ``window`` (N, W)), one window per row.
    """
    assert logits.ndim in (1, 2), f"Got {logits.shape}"
    assert window.ndim == logits.ndim, f"Got {window.shape} vs {logits.shape}"
    assert is_penalizable.shape == logits.shape[-1:], (
        f"Got {is_penalizable.shape} vs {logits.shape}"
    )
    if penalty == 1.0 or window.numel() == 0:
        return logits

    single = logits.ndim == 1
    batch_logits = logits.unsqueeze(0) if single else logits
    batch_window = window.unsqueeze(0) if single else window
    seen = torch.zeros_like(batch_logits, dtype=torch.bool)
    seen.scatter_(1, batch_window, True)
    seen &= is_penalizable
    penalized = torch.where(
        batch_logits > 0, batch_logits / penalty, batch_logits * penalty
    )
    out = torch.where(seen, penalized, batch_logits)
    return out.squeeze(0) if single else out


@torch.no_grad()
def generate_batch(
    model: MusicTransformer,
    tokenizer: REMI,
    device: torch.device,
    length: int,
    temperature: float = 0.92,
    composer: str = "Frédéric Chopin",
    min_p: float = 0.02,
    cfg_scale: float = 1.2,
    repetition_penalty: float = 1.15,
    penalty_window: int = 64,
    min_new_tokens: int = 96,
    use_kv_cache: bool = True,
    seed: int | None = None,
    prefix_ids: list[int] | None = None,
    num_samples: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[list[int]]:
    """Sample *num_samples* independent pieces in one batched pass.

    Every sequence is ``[composer, *prefix_ids, *new_tokens]``; *length* counts
    only the new tokens. *prefix_ids* is typically a primer from
    :func:`musiclm.inference.priming.load_primer`.
    ``progress_callback(done, total)`` is called as new tokens are produced.
    """
    assert temperature > 0.0, f"Got temperature={temperature}"
    assert 0.0 <= min_p < 1.0, f"Got min_p={min_p}"
    assert repetition_penalty >= 1.0, f"Got repetition_penalty={repetition_penalty}"
    assert penalty_window > 0, f"Got penalty_window={penalty_window}"
    assert num_samples >= 1, f"Got num_samples={num_samples}"

    if seed is not None:
        # Seeding the global generators keeps RNG consumption identical to the
        # unseeded code path, so seeded runs are reproducible bit-for-bit.
        torch.manual_seed(seed)

    categories = TokenCategories(tokenizer, device)
    start_token = pick_start_token(tokenizer, composer)
    prefix_ids = list(prefix_ids or [])

    uncond_token = pick_unconditional_token(tokenizer)
    if cfg_scale != 1.0 and uncond_token is None:
        logger.warning(
            "Tokenizer has no %s token, disabling classifier-free guidance. "
            "Re-run `musiclm-tokenize` to rebuild the vocabulary.",
            composer_vocab_token(UNCONDITIONAL_COMPOSER),
        )
        cfg_scale = 1.0
    use_cfg = cfg_scale != 1.0

    prompt_len = 1 + len(prefix_ids)
    if prompt_len >= model.max_seq_len:
        raise ValueError(
            f"Prompt of {prompt_len} tokens leaves no room under "
            f"max_seq_len={model.max_seq_len}; use a shorter primer"
        )
    # The rotary cache is sized to max_seq_len, so generation cannot outrun it.
    max_tokens = min(prompt_len + length, model.max_seq_len)
    if max_tokens < prompt_len + length:
        logger.warning(
            "Requested %d new tokens after a %d-token prompt but max_seq_len=%d; "
            "truncating the request",
            length,
            prompt_len,
            model.max_seq_len,
        )
    new_total = max_tokens - prompt_len

    # Rows [0, N) are conditional; rows [N, 2N) are their unconditional twins.
    copies = 2 if use_cfg else 1
    generated = torch.empty(
        (copies * num_samples, max_tokens), device=device, dtype=torch.long
    )
    generated[:num_samples, 0] = start_token
    if use_cfg:
        assert uncond_token is not None
        generated[num_samples:, 0] = uncond_token
        logger.info(
            "Classifier-free guidance scale=%.2f (uncond id=%d)",
            cfg_scale,
            uncond_token,
        )
    if prefix_ids:
        generated[:, 1:prompt_len] = torch.tensor(
            prefix_ids, device=device, dtype=torch.long
        )

    past_key_values: list[KVCache | None] | None = None
    log_every = max(new_total // 10, 1)
    progress_every = max(new_total // 100, 1)
    lengths = torch.full((num_samples,), max_tokens, device=device, dtype=torch.long)
    finished = torch.zeros(num_samples, device=device, dtype=torch.bool)

    step = prompt_len
    while step < max_tokens:
        if use_kv_cache:
            context = (
                generated[:, :step]
                if past_key_values is None
                else generated[:, step - 1 : step]
            )
            logits, past_key_values = model(
                context,
                past_key_values=past_key_values,
                use_cache=True,
            )
        else:
            logits, _ = model(generated[:, :step])

        conditional = logits[:num_samples, -1, :].float()
        if use_cfg:
            unconditional = logits[num_samples:, -1, :].float()
            next_logits = unconditional + cfg_scale * (conditional - unconditional)
        else:
            next_logits = conditional.clone()
        assert next_logits.shape == (num_samples, model.vocab_size), (
            f"Got {next_logits.shape}"
        )

        new_so_far = step - prompt_len
        next_logits[:, categories.forbidden_ids] = float("-inf")
        if new_so_far < min_new_tokens:
            next_logits[:, categories.eos_token_id] = float("-inf")

        next_logits = apply_repetition_penalty(
            next_logits,
            generated[:num_samples, max(0, step - penalty_window) : step],
            repetition_penalty,
            categories.is_pitch,
        )
        next_logits = min_p_filter(next_logits / temperature, min_p)
        probs = F.softmax(next_logits, dim=-1)
        # Kept on device: writing the sample straight into `generated` avoids a
        # host synchronization on every step of the loop.
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        generated[:, step] = next_token.repeat(copies)
        step += 1
        new_so_far += 1

        # EOS needs host-side values, so it only synchronizes once EOS is
        # actually reachable.
        if new_so_far > min_new_tokens:
            just_ended = next_token.eq(categories.eos_token_id) & ~finished
            lengths = torch.where(just_ended, step - 1, lengths)
            finished |= just_ended
            if bool(finished.all()):
                logger.info("EOS sampled in every sample by step %d, stopping", step - 1)
                break

        if progress_callback is not None and (
            new_so_far % progress_every == 0 or step == max_tokens
        ):
            progress_callback(new_so_far, new_total)
        if new_so_far % log_every == 0 or step == max_tokens:
            logger.info("Generated %d/%d new tokens", new_so_far, new_total)

    rows = generated[:num_samples].tolist()
    return [row[:end] for row, end in zip(rows, lengths.tolist())]


def generate(
    model: MusicTransformer,
    tokenizer: REMI,
    device: torch.device,
    length: int,
    **kwargs: Any,
) -> list[int]:
    """Single-sample :func:`generate_batch`; returns one token sequence."""
    assert "num_samples" not in kwargs, "Use generate_batch for several samples"
    return generate_batch(model, tokenizer, device, length, **kwargs)[0]


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    # One score per candidate (lower is closer to real music); empty for N=1.
    scores: list[float]
    chosen: int
    primer_tokens: int


def generate_piece(
    model: MusicTransformer,
    tokenizer: REMI,
    device: torch.device,
    length: int,
    *,
    composer: str,
    prime_midi: str | Path | None = None,
    prime_bars: int = 4,
    keep_primer: bool = False,
    num_candidates: int = 1,
    reference_stats: dict[str, Any] | None = None,
    **sampling: Any,
) -> GenerationResult:
    """Prime, sample *num_candidates* pieces, keep the most realistic one.

    With *prime_midi*, the first *prime_bars* bars of that file seed the model
    and are cut from the result unless *keep_primer* is set.
    """
    prefix = load_primer(tokenizer, prime_midi, prime_bars) if prime_midi else None
    primer_len = len(prefix) if prefix else 0
    if primer_len:
        logger.info("Primed with %d bars of %s (%d tokens)", prime_bars, prime_midi, primer_len)

    candidates = generate_batch(
        model,
        tokenizer,
        device,
        length,
        composer=composer,
        prefix_ids=prefix,
        num_samples=num_candidates,
        **sampling,
    )
    scores: list[float] = []
    chosen = 0
    if num_candidates > 1:
        scores = score_candidates(
            tokenizer, candidates, composer, reference_stats, primer_len
        )
        chosen = min(range(len(scores)), key=scores.__getitem__)
        logger.info("Kept candidate %d of %d", chosen + 1, num_candidates)

    token_ids = candidates[chosen]
    if primer_len and not keep_primer:
        token_ids = drop_primer(token_ids, primer_len, tokenizer)
    return GenerationResult(token_ids, scores, chosen, primer_len)


def tokens_to_midi_file(
    tokenizer: REMI,
    token_ids: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composer_ids = {
        learned_token_id(tokenizer, token)
        for token in list_composer_tokens(tokenizer)
    }
    eos_id = learned_token_id(tokenizer, "EOS_None")
    decode_ids = [
        token_id
        for token_id in token_ids
        if token_id not in composer_ids and token_id != eos_id
    ]
    sequence = TokSequence(ids=decode_ids, are_ids_encoded=tokenizer.is_trained)
    score = tokenizer.decode([sequence])
    score.dump_midi(output_path)
    logger.info("Saved generated MIDI to %s", output_path)


def main(cfg: GenerateConfig | None = None) -> Path:
    """Generate one piece and save it as MIDI; returns the output path."""
    if cfg is None:
        cfg = GenerateConfig()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = get_device()
    logger.info("Device: %s", device)

    tokenizer = load_tokenizer(cfg.tokenizer_path)
    model = load_model(cfg.checkpoint_path, device)
    if model.vocab_size != len(tokenizer):
        logger.warning(
            "Checkpoint vocab_size=%d != tokenizer vocab_size=%d; using the "
            "checkpoint value to stay consistent with the trained weights",
            model.vocab_size,
            len(tokenizer),
        )

    result = generate_piece(
        model,
        tokenizer,
        device,
        cfg.length,
        composer=cfg.composer,
        prime_midi=cfg.prime_midi,
        prime_bars=cfg.prime_bars,
        keep_primer=cfg.keep_primer,
        num_candidates=cfg.num_candidates,
        reference_stats=load_reference_stats() if cfg.num_candidates > 1 else None,
        temperature=cfg.temperature,
        min_p=cfg.min_p,
        cfg_scale=cfg.cfg_scale,
        repetition_penalty=cfg.repetition_penalty,
        penalty_window=cfg.penalty_window,
        min_new_tokens=cfg.min_new_tokens,
        use_kv_cache=cfg.use_kv_cache,
        seed=cfg.seed,
    )
    logger.info("Generated %d tokens", len(result.token_ids))

    tokens_to_midi_file(tokenizer, result.token_ids, cfg.output_path)
    return Path(cfg.output_path)


if __name__ == "__main__":
    main()
