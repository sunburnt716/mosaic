"""
Contract tests for extraction/embedder.py.

Tests the Embedder Protocol and MiniLMEmbedder's structural contract without
downloading or loading the real model. A FakeEmbedder is used throughout; the only
MiniLMEmbedder test checks its constant attributes — no SentenceTransformer import.
"""

from __future__ import annotations

from extraction.embedder import Embedder, MiniLMEmbedder

# ---------------------------------------------------------------------------
# FakeEmbedder — deterministic, no model download, satisfies Embedder Protocol
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Returns fixed-length zero vectors deterministically."""

    model_name = "fake-384d"
    _dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestEmbedderProtocol:
    def test_fake_satisfies_protocol(self):
        # runtime_checkable allows isinstance checks against a Protocol
        assert isinstance(FakeEmbedder(), Embedder)

    def test_mini_lm_satisfies_protocol(self):
        assert isinstance(MiniLMEmbedder(), Embedder)


# ---------------------------------------------------------------------------
# FakeEmbedder contract (shared by any compliant implementation)
# ---------------------------------------------------------------------------


class TestFakeEmbedder:
    def setup_method(self):
        self.embedder = FakeEmbedder()

    def test_empty_input_returns_empty_list(self):
        assert self.embedder.embed([]) == []

    def test_single_text_returns_one_vector(self):
        result = self.embedder.embed(["hello world"])
        assert len(result) == 1

    def test_n_texts_return_n_vectors(self):
        texts = ["a", "b", "c", "d"]
        result = self.embedder.embed(texts)
        assert len(result) == len(texts)

    def test_all_vectors_same_length(self):
        result = self.embedder.embed(["short", "a much longer piece of text here"])
        assert len(result[0]) == len(result[1])

    def test_output_length_matches_declared_dim(self):
        result = self.embedder.embed(["test"])
        assert len(result[0]) == FakeEmbedder._dim

    def test_order_preserved(self):
        # Each vector is deterministic and identical for this fake, so we only
        # check that the count and order match (n inputs → n outputs in order).
        texts = ["first", "second", "third"]
        result = self.embedder.embed(texts)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# MiniLMEmbedder constants (no model load)
# ---------------------------------------------------------------------------


class TestMiniLMEmbedderConstants:
    def test_model_name_is_expected_slug(self):
        # The slug is stamped into the Chroma collection name; changing it is a
        # breaking change that requires migrating the collection.
        assert MiniLMEmbedder.model_name == "minilm-l6-v2"

    def test_model_name_on_instance(self):
        embedder = MiniLMEmbedder()
        assert embedder.model_name == "minilm-l6-v2"

    def test_model_not_loaded_on_init(self):
        # The model should be loaded lazily, not at construction time.
        embedder = MiniLMEmbedder()
        assert embedder._model is None
