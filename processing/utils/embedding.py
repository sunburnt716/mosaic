"""
Shared MiniLM embedding — the one embedding model, loaded once.

CLAUDE.md's collection invariant ("one embedding model per Chroma collection — never mix
models") extends to the client side: the corpus embedder (Phase 2, not yet built) and the
retrieval query embedder (Retrieval Pipeline Phase 1) must use the exact same model, or
cosine similarity between query and chunk vectors is meaningless. This module is the one
place that model is named, mirroring `processing.utils.tokenization`'s lazy-load pattern for
the same reason (and the same model name — MiniLM's tokenizer and embedder are paired).

The model is loaded lazily and cached at module level: the heavy `sentence-transformers`
import only happens on first real use, so the offline unit suite — which monkeypatches
`_model` with a lightweight fake — never pulls it in (mirrors tokenization.py and the
ingestion adapters' lazy `requests`/`feedparser` imports).

Exports:
  embed_text(text) -> list[float]   MiniLM embedding of a single string
"""

from __future__ import annotations

# One embedding model per collection — must match the tokenizer used for chunk sizing.
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazily populated cache. None until the first real embedding; tests overwrite it with a
# fake so the model is never downloaded offline.
_model = None


def _get_model():
    """Return the cached MiniLM embedding model, loading it once on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single string with MiniLM, returning a plain list of floats."""
    model = _get_model()
    return list(model.encode(text))
