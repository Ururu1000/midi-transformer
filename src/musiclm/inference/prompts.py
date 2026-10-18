"""Free-text prompt parsing: human words -> generation parameters.

The generative model itself never sees raw text. This module maps a phrase
like "сгенерируй мне шопена, подлиннее" onto the existing conditioning
knobs (``composer`` / ``length`` / ``temperature``), which the caller then
passes to ``musiclm.inference.sampler.generate`` exactly as if they came
from CLI flags or UI sliders. No retraining involved.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Russian aliases (including common inflections) keyed by normalized latin
# last name. Latin forms are matched generically; this table only adds
# cyrillic variants that normalization alone cannot produce.
RU_ALIASES: dict[str, list[str]] = {
    "chopin": ["шопен", "шопена", "шопену", "шопеном", "шопене"],
    "schubert": ["шуберт", "шуберта", "шуберту", "шубертом"],
    "bach": ["бах", "баха", "баху", "бахом", "бахе"],
    "liszt": ["лист", "листа", "листу", "листом", "листе"],
    "beethoven": ["бетховен", "бетховена", "бетховену", "бетховеном"],
    "scarlatti": ["скарлатти"],
    "mozart": ["моцарт", "моцарта", "моцарту", "моцартом"],
    "haydn": ["гайдн", "гайдна", "гайдну", "гайдном"],
    "schumann": ["шуман", "шумана", "шуману", "шуманом"],
    "rachmaninoff": ["рахманинов", "рахманинова", "рахманинову", "рахманиновым"],
    "debussy": ["дебюсси", "дебуси"],
    "handel": ["гендель", "генделя", "генделю", "генделем"],
    "scriabin": ["скрябин", "скрябина", "скрябину", "скрябиным"],
}

MIN_LENGTH = 256
MAX_LENGTH = 4096
SHORT_LENGTH = 512
DEFAULT_LENGTH = 1024
LONG_LENGTH = 2048
EXPERIMENTAL_TEMPERATURE = 1.05
STRICT_TEMPERATURE = 0.85


@dataclass(frozen=True)
class ParsedPrompt:
    """Result of parsing; ``None`` fields mean "keep the current setting"."""

    composer: str | None
    length: int | None
    temperature: float | None
    interpretation: str


def _normalize(text: str) -> str:
    lowered = text.lower().replace("ё", "е")
    stripped = "".join(
        char for char in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(char)
    )
    cleaned = re.sub(r"[^a-zа-я0-9]+", " ", stripped)
    return re.sub(r"\s+", " ", cleaned).strip()


def _composer_keys(canonical: str) -> set[str]:
    """Matchable keys for one canonical composer name."""
    keys = {_normalize(canonical)}
    if "," in canonical:
        last = canonical.split(",")[0].strip()
    else:
        last = canonical.split()[-1]
    norm_last = _normalize(last)
    keys.add(norm_last)
    for alias in RU_ALIASES.get(norm_last, []):
        keys.add(_normalize(alias))
    return {key for key in keys if key}


def _match_composer(norm_prompt: str, available: list[str]) -> str | None:
    padded = f" {norm_prompt} "
    best: str | None = None
    best_len = 0
    for canonical in available:
        for key in _composer_keys(canonical):
            hit = key in padded if " " in key else f" {key} " in padded
            if hit and len(key) > best_len:
                best, best_len = canonical, len(key)
    return best


def _match_length(norm_prompt: str) -> int | None:
    explicit = re.search(r"(\d{3,4})\s*(токен|token)", norm_prompt)
    if explicit:
        return max(MIN_LENGTH, min(MAX_LENGTH, int(explicit.group(1))))
    if any(
        stem in norm_prompt
        for stem in (
            "две минут",
            "2 минут",
            "пару минут",
            "two minut",
            "2 minut",
            "couple of minut",
        )
    ):
        return LONG_LENGTH
    if "minut" in norm_prompt and "token" not in norm_prompt:
        # "a minute piece", "one minute", "на минуту" — roughly the default.
        return DEFAULT_LENGTH
    if any(
        stem in norm_prompt
        for stem in (
            "длинн",
            "подлиннее",
            "подлинней",
            "побольше",
            "long",
            "longer",
            "lengthy",
            "extended",
        )
    ):
        return LONG_LENGTH
    if any(
        stem in norm_prompt
        for stem in ("коротк", "покороче", "short", "brief", "tiny")
    ):
        return SHORT_LENGTH
    if any(
        stem in norm_prompt
        for stem in ("средн", "обычн", "medium", "average", "normal")
    ):
        return DEFAULT_LENGTH
    return None


def _match_temperature(norm_prompt: str) -> float | None:
    if any(
        stem in norm_prompt
        for stem in (
            "эксперимент", "странн", "необычн", "креатив", "свобод",
            "experimental", "weird", "creative", "random", "wild",
            "adventurous", "surprising",
        )
    ):
        return EXPERIMENTAL_TEMPERATURE
    if any(
        stem in norm_prompt
        for stem in (
            "строг", "точн", "аккурат",
            "strict", "precise", "conservative", "careful", "faithful",
        )
    ):
        return STRICT_TEMPERATURE
    return None


def parse_prompt(text: str, available_composers: list[str]) -> ParsedPrompt:
    """Map free text onto generation params; ``None`` = keep current value."""
    if not text or not text.strip():
        return ParsedPrompt(
            composer=None,
            length=None,
            temperature=None,
            interpretation="Пустой запрос — использую текущие настройки.",
        )
    norm = _normalize(text)
    composer = _match_composer(norm, available_composers) if norm else None
    length = _match_length(norm)
    temperature = _match_temperature(norm)

    parts = [
        f"Композитор: {composer}"
        if composer
        else "Композитор: не распознан (оставлю текущий)",
        f"Длина: {length} токенов" if length else "Длина: текущая",
        f"Температура: {temperature}" if temperature else "Температура: текущая",
    ]
    return ParsedPrompt(
        composer=composer,
        length=length,
        temperature=temperature,
        interpretation="Понял так: " + ", ".join(parts) + ".",
    )


def load_available_composers(
    mapping_path: Path | None = None,
    tokenizer_path: Path | None = None,
) -> list[str]:
    """Composer display names, preferring the mapping file over the tokenizer."""
    from musiclm.config import COMPOSER_MAPPING_PATH, TOKENIZER_PATH

    mapping = Path(mapping_path) if mapping_path else COMPOSER_MAPPING_PATH
    if mapping.exists():
        with open(mapping) as fh:
            return list(json.load(fh).get("top_composers", []))
    tok_path = Path(tokenizer_path) if tokenizer_path else TOKENIZER_PATH
    if not tok_path.exists():
        return []
    from musiclm.data.tokenizer import ComposerREMI, list_composer_tokens

    tokenizer = ComposerREMI(params=str(tok_path))
    return [
        token.removeprefix("Composer_").replace("_", " ")
        for token in list_composer_tokens(tokenizer)
    ]
