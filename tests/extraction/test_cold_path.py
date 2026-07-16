"""
Contract tests for extraction/cold_path.py — ensure_processed().

Uses FakeEmbedder (deterministic, no model download) and chromadb.EphemeralClient()
(in-memory), matching the pattern in test_extraction_engine.py, plus an in-memory
RawStore so the suite is fully offline.

Key contracts verified:
  - Missing doc_id returns False.
  - An already-processed document is a no-op (returns True, embedder never called).
  - An unprocessed document gets extracted and its status flips to "processed".
  - An extraction error returns False and leaves status unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chromadb", reason="chromadb not installed")

import chromadb

from extraction.chroma_store import ChromaVectorStore
from extraction.cold_path import ensure_processed
from ingestion.storage.raw_store import RawStore
from tests.extraction.fixtures import ARTICLE_BODY, make_document

_DIM = 384
_SETTINGS = chromadb.Settings(allow_reset=True)


class FakeEmbedder:
    model_name = "fake-384d"

    def __init__(self):
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1] * _DIM for _ in texts]


def _fresh_client() -> chromadb.ClientAPI:
    client = chromadb.EphemeralClient(settings=_SETTINGS)
    client.reset()
    return client


class TestEnsureProcessed:
    def setup_method(self):
        self.client = _fresh_client()
        self.embedder = FakeEmbedder()
        self.store = ChromaVectorStore(self.client, self.embedder.model_name)
        self.raw = RawStore(":memory:")

    def teardown_method(self):
        self.raw.close()

    def test_missing_doc_id_returns_false(self):
        result = ensure_processed("does-not-exist", self.raw, self.embedder, self.store)
        assert result is False

    def test_already_processed_document_is_noop(self):
        doc = make_document(id="doc-1", body=ARTICLE_BODY, status="processed")
        self.raw.save_document(doc)

        result = ensure_processed("doc-1", self.raw, self.embedder, self.store)

        assert result is True
        assert self.embedder.calls == 0

    def test_unprocessed_document_is_extracted_and_marked_processed(self):
        doc = make_document(id="doc-1", body=ARTICLE_BODY, status="unprocessed")
        self.raw.save_document(doc)

        result = ensure_processed("doc-1", self.raw, self.embedder, self.store)

        assert result is True
        assert self.embedder.calls == 1
        assert self.raw.get_document("doc-1").status == "processed"

    def test_extraction_error_returns_false_and_status_unchanged(self):
        doc = make_document(id="doc-1", body=ARTICLE_BODY, status="unprocessed")
        self.raw.save_document(doc)

        class _FailingEmbedder:
            model_name = "fake-384d"

            def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("simulated embedding failure")

        result = ensure_processed("doc-1", self.raw, _FailingEmbedder(), self.store)

        assert result is False
        assert self.raw.get_document("doc-1").status == "unprocessed"
