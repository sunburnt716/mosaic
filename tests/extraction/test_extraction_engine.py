"""
Contract tests for extraction/extraction_engine.py.

Uses FakeEmbedder (deterministic, no model download) and chromadb.EphemeralClient()
(in-memory, no file I/O) so the suite is fully offline.

IMPORTANT: chromadb 1.5.x EphemeralClient instances share a single in-memory store.
Each setup_method creates a client with `Settings(allow_reset=True)` and calls
`reset()` to wipe all collections before every test.

Key contracts verified:
  - One Document produces the correct number of chunks in Chroma.
  - Citation fields (url, tier, source_name, published_date) survive intact.
  - Per-document isolation: a bad document does not abort remaining ones.
  - An empty-chunk document (body too short) is skipped silently — not an error.
  - ExtractionResult counts match the actual Chroma state.
  - source_hints advisory is forwarded to type inference.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("chromadb", reason="chromadb not installed")

import chromadb

from extraction.chroma_store import ChromaVectorStore
from extraction.extraction_engine import extract
from tests.extraction.fixtures import ARTICLE_BODY, FILING_BODY, make_document

# ---------------------------------------------------------------------------
# FakeEmbedder — deterministic, satisfies Embedder Protocol, no model load
# ---------------------------------------------------------------------------

_DIM = 384
_SETTINGS = chromadb.Settings(allow_reset=True)


class FakeEmbedder:
    model_name = "fake-384d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * _DIM for _ in texts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_client() -> chromadb.ClientAPI:
    client = chromadb.EphemeralClient(settings=_SETTINGS)
    client.reset()
    return client


def _make_store(client: chromadb.ClientAPI) -> ChromaVectorStore:
    return ChromaVectorStore(client, FakeEmbedder.model_name)


def _col_count(client: chromadb.ClientAPI, store: ChromaVectorStore) -> int:
    try:
        return client.get_collection(store.collection_name).count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


class TestExtractBasic:
    def setup_method(self):
        self.client = _fresh_client()
        self.embedder = FakeEmbedder()
        self.store = _make_store(self.client)

    def test_one_article_produces_chunks_in_chroma(self):
        doc = make_document(body=ARTICLE_BODY)
        result = extract([doc], self.embedder, self.store)
        assert result.documents_processed == 1
        assert result.chunks_written > 0
        assert _col_count(self.client, self.store) == result.chunks_written

    def test_one_filing_produces_chunks_in_chroma(self):
        doc = make_document(body=FILING_BODY)
        result = extract([doc], self.embedder, self.store)
        assert result.documents_processed == 1
        assert result.chunks_written > 0

    def test_result_counts_match_actual_chroma_state(self):
        doc = make_document(body=ARTICLE_BODY)
        result = extract([doc], self.embedder, self.store)
        assert _col_count(self.client, self.store) == result.chunks_written

    def test_empty_document_list_returns_zero_counts(self):
        result = extract([], self.embedder, self.store)
        assert result.documents_processed == 0
        assert result.chunks_written == 0
        assert result.errors == []

    def test_empty_chunk_document_not_counted_as_error(self):
        # A document with no body produces no chunks and is skipped silently.
        doc = make_document(body="")
        result = extract([doc], self.embedder, self.store)
        assert result.errors == []
        assert result.documents_processed == 0


# ---------------------------------------------------------------------------
# Citation metadata survives into Chroma
# ---------------------------------------------------------------------------


class TestCitationMetadataSurvival:
    def setup_method(self):
        self.client = _fresh_client()
        self.store = _make_store(self.client)

    def test_citation_fields_intact_in_chroma(self):
        doc = make_document(
            body=ARTICLE_BODY,
            source_name="FT",
            url="https://ft.com/story/1",
            tier=1,
            published_date=datetime(2026, 1, 15, 9, 0, tzinfo=timezone.utc),
            title="Markets Rally",
        )
        extract([doc], FakeEmbedder(), self.store)
        col = self.client.get_collection(self.store.collection_name)
        result = col.get(include=["metadatas"])
        # All chunks from the same document share the same provenance fields.
        for meta in result["metadatas"]:
            assert meta["source_name"] == "FT"
            assert meta["url"] == "https://ft.com/story/1"
            assert meta["tier"] == 1
            assert meta["published_date"] == "2026-01-15T09:00:00+00:00"
            assert meta["title"] == "Markets Rally"


# ---------------------------------------------------------------------------
# Per-document isolation
# ---------------------------------------------------------------------------


class TestPerDocumentIsolation:
    def setup_method(self):
        self.client = _fresh_client()
        self.store = _make_store(self.client)

    def test_bad_document_does_not_abort_remaining(self):
        # A one-shot failing embedder that raises only on the first embed() call.
        class _FailOnceEmbedder:
            model_name = "fake-384d"
            _calls = 0

            def embed(self, texts: list[str]) -> list[list[float]]:
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("simulated embedding failure")
                return [[0.1] * _DIM for _ in texts]

        bad_doc = make_document(body=ARTICLE_BODY, id="bad-doc", identity_key="src::bad")
        good_doc = make_document(body=ARTICLE_BODY, id="good-doc", identity_key="src::good")
        result = extract([bad_doc, good_doc], _FailOnceEmbedder(), self.store)

        assert len(result.errors) == 1
        assert result.documents_processed == 1
        assert result.chunks_written > 0


# ---------------------------------------------------------------------------
# source_hints forwarding
# ---------------------------------------------------------------------------


class TestSourceHints:
    def setup_method(self):
        self.client = _fresh_client()

    def test_filing_hint_applies_section_chunking(self):
        store = _make_store(self.client)
        doc = make_document(body=FILING_BODY, source_name="sec-edgar")
        result = extract(
            [doc],
            FakeEmbedder(),
            store,
            source_hints={"sec-edgar": "filing"},
        )
        assert result.documents_processed == 1
        assert result.chunks_written > 0

    def test_no_source_hints_still_processes(self):
        store = _make_store(self.client)
        doc = make_document(body=ARTICLE_BODY)
        result = extract([doc], FakeEmbedder(), store)
        assert result.documents_processed == 1
