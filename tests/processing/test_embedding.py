"""
Contract tests for processing/utils/embedding.py.

Pins the public surface against the injected FakeEmbeddingModel so the suite stays offline.
Assertions target the contract (fixed-length float list, deterministic, lazy-loaded), not
MiniLM's specific vector values.
"""

from __future__ import annotations


class TestEmbedText:
    def test_returns_list_of_floats(self, fake_embedder):
        from processing.utils.embedding import embed_text

        vector = embed_text("alpha beta gamma")
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)

    def test_deterministic_for_same_text(self, fake_embedder):
        from processing.utils.embedding import embed_text

        assert embed_text("same query") == embed_text("same query")

    def test_different_text_yields_different_vector(self, fake_embedder):
        from processing.utils.embedding import embed_text

        assert embed_text("NVDA earnings") != embed_text("completely different text")

    def test_empty_text_still_returns_fixed_length_vector(self, fake_embedder):
        from processing.utils.embedding import embed_text

        assert len(embed_text("")) == len(embed_text("non-empty"))


class TestLazyLoading:
    def test_get_model_returns_injected_fake_without_sentence_transformers(self, fake_embedder):
        # The fixture populated the cache, so _get_model must not attempt an import.
        from processing.utils import embedding

        assert embedding._get_model() is fake_embedder
