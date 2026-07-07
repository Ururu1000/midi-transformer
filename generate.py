from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from miditok import REMI, TokSequence
from torch import Tensor

from model import MusicTransformer

CHECKPOINT_PATH = Path("checkpoints/model_epoch_12.pt")
TOKENIZER_PATH = Path("data/processed/tokenizer.json")
OUTPUT_PATH = Path("data/processed/output.mid")

GENERATION_LENGTH = 512
TEMPERATURE = 1.0
TOP_K = 65

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device) -> MusicTransformer:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = int(checkpoint["vocab_size"])
    model = MusicTransformer(vocab_size=vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        "Loaded model from %s | vocab_size=%d | %.2fM params",
        checkpoint_path,
        vocab_size,
        model.get_num_params(),
    )
    return model


def pick_start_token(tokenizer: REMI) -> int:
    if "BOS_None" in tokenizer.vocab:
        start_token = tokenizer["BOS_None"]
        logger.info("Using BOS token as seed (id=%d)", start_token)
        return start_token

    start_token = int(torch.randint(0, len(tokenizer), (1,)).item())
    logger.info("No BOS token, using random seed token (id=%d)", start_token)
    return start_token


def sample_next_token(
    logits: Tensor,
    temperature: float,
    top_k: int,
    forbidden_token_ids: Tensor,
) -> int:
    assert logits.ndim == 1, f"Got {logits.shape}"

    logits = logits.clone()
    logits[forbidden_token_ids] = float("-inf")
    logits = logits / temperature

    top_k = min(top_k, logits.shape[-1])
    top_values, top_indices = torch.topk(logits, top_k)
    probs = F.softmax(top_values, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return int(top_indices[sampled].item())


@torch.no_grad()
def generate(
    model: MusicTransformer,
    tokenizer: REMI,
    device: torch.device,
    length: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    start_token = pick_start_token(tokenizer)
    generated = [start_token]

    forbidden = [
        tokenizer[token]
        for token in ("PAD_None", "BOS_None", "EOS_None", "MASK_None")
        if token in tokenizer.vocab
    ]
    forbidden_token_ids = torch.tensor(forbidden, dtype=torch.long, device=device)

    log_every = max(length // 10, 1)
    for step in range(1, length):
        context = torch.tensor([generated], dtype=torch.long, device=device)
        context = context[:, -model.max_seq_len :]

        logits = model(context)
        next_logits = logits[0, -1, :]
        next_token = sample_next_token(
            next_logits, temperature, top_k, forbidden_token_ids
        )
        generated.append(next_token)

        if step % log_every == 0 or step == length - 1:
            logger.info("Generating token %d/%d (id=%d)", step + 1, length, next_token)

    assert len(generated) == length, f"Got {len(generated)}"
    return generated


def tokens_to_midi_file(
    tokenizer: REMI,
    token_ids: list[int],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sequence = TokSequence(ids=token_ids)
    score = tokenizer.tokens_to_midi([sequence])
    score.dump_midi(output_path)
    logger.info("Saved generated MIDI to %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = REMI(params=str(TOKENIZER_PATH))
    logger.info("Loaded tokenizer from %s | vocab_size=%d", TOKENIZER_PATH, len(tokenizer))

    model = load_model(CHECKPOINT_PATH, device)

    token_ids = generate(
        model,
        tokenizer,
        device,
        GENERATION_LENGTH,
        TEMPERATURE,
        TOP_K,
    )
    logger.info("Generated %d tokens", len(token_ids))

    tokens_to_midi_file(tokenizer, token_ids, OUTPUT_PATH)


if __name__ == "__main__":
    main()
