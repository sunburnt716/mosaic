"""
Phase 2 embedding — the boundary between Chunks and vectors.

`Embedder` is a Protocol so tests inject a `FakeEmbedder` without downloading any model.
`MiniLMEmbedder` is the concrete implementation: lazy-loads `SentenceTransformer` on first
call and caches it, matching the lazy-load pattern used by `extraction/utils/tokenization.py`.

The `model_name` attribute is the canonical slug stamped onto the Chroma collection name.
It enforces the one-model-per-collection rule: two embedders with different `model_name`
values must never write to the same collection.

Public surface:
  Embedder          — Protocol for DI/testing
  MiniLMEmbedder    — sentence-transformers/all-MiniLM-L6-v2, normalized 384-d vectors
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass

_MINILM_HF_ID = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """Contract for any embedding backend used by the extraction pipeline."""

    # Canonical slug written into the Chroma collection name; immutable once a
    # collection is created — changing it without migrating the collection breaks retrieval.
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order.

        Empty input returns an empty list. All vectors in the output must have the same
        dimensionality. Callers must not mutate the returned lists.
        """
        ...


class MiniLMEmbedder:
    """sentence-transformers/all-MiniLM-L6-v2 — 384-d L2-normalised vectors.

    The model is loaded lazily on the first `embed()` call so import time stays fast
    and CI runs that inject a FakeEmbedder never touch the network or disk.
    """

    model_name = "minilm-l6-v2"

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        # Deferred import: sentence_transformers is only required at embed time, not at
        # import time. This keeps the extraction package importable in CI without the dep.
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        if self._model is None:
            self._model = SentenceTransformer(_MINILM_HF_ID)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        # convert_to_numpy=True returns a 2-D ndarray; .tolist() yields the nested
        # Python float lists Chroma accepts. normalize_embeddings=True L2-normalises each
        # vector so the collection's cosine space (see chroma_store.py) gets the unit
        # vectors it assumes — all-MiniLM-L6-v2 does not normalise by default.
        return model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()
