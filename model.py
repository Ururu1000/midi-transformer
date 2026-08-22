"""Deprecated shim — the model lives in musiclm.model; kept for old imports."""
from musiclm.model import (  # noqa: F401
    KVCache,
    MusicTransformer,
    build_document_causal_mask,
    get_device,
)
