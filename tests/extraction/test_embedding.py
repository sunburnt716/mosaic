"""
Contract tests for extraction/utils/embedding.py.

Pins the public surface against the injected FakeEmbeddingModel so the suite stays offline.
Assertions target the contract (fixed-length float list, deterministic, lazy-loaded), not
MiniLM's specific vector values.
"""

from __future__ import annotations


class TestEmbedText:
    def test_returns_list_of_floats(self, fake_embedder):
        from extraction.utils.embedding import embed_text

        vector = embed_text("alpha beta gamma")
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)

    def test_deterministic_for_same_text(self, fake_embedder):
        from extraction.utils.embedding import embed_text

        assert embed_text("same query") == embed_text("same query")

    def test_different_text_yields_different_vector(self, fake_embedder):
        from extraction.utils.embedding import embed_text

        assert embed_text("NVDA earnings") != embed_text("completely different text")

    def test_empty_text_still_returns_fixed_length_vector(self, fake_embedder):
        from extraction.utils.embedding import embed_text

        assert len(embed_text("")) == len(embed_text("non-empty"))


class _NumpyFloat32Like:
    """Stands in for numpy.float32: not a plain float, but compares/converts like one."""

    def __init__(self, value: float) -> None:
        self._value = value

    def __float__(self) -> float:
        return self._value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and self._value == other


class _NumpyArrayLike:
    """Mimics the real gap between `list(ndarray)` and `ndarray.tolist()`.

    `list(x)` yields the numpy-scalar wrapper (not a plain float) for each element,
    exactly like a real numpy array — this is what silently broke embed_text before
    `.tolist()` was used, since chromadb's real query_embeddings validation rejects
    numpy scalar types but the offline suite's plain-float fake never triggered it.
    """

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __iter__(self):
        return iter(_NumpyFloat32Like(v) for v in self._values)

    def tolist(self) -> list[float]:
        return list(self._values)


class TestNumpyReturnRegression:
    def test_embed_text_uses_tolist_not_plain_list(self, monkeypatch):
        # Regression test: embed_text must call .tolist() on the model's output, not
        # list(...) — the latter passed this file's other tests (which use a fake
        # returning plain floats already) while still being broken against a real model.
        from extraction.utils import embedding

        class _FakeModel:
            def encode(self, text: str):
                return _NumpyArrayLike([0.1, 0.2, 0.3])

        monkeypatch.setattr(embedding, "_model", _FakeModel())
        vector = embedding.embed_text("alpha")
        assert all(type(x) is float for x in vector)


class TestLazyLoading:
    def test_get_model_returns_injected_fake_without_sentence_transformers(self, fake_embedder):
        # The fixture populated the cache, so _get_model must not attempt an import.
        from extraction.utils import embedding

        assert embedding._get_model() is fake_embedder
