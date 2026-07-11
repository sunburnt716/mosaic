"""
Contract tests for extraction/chroma_store.py.

Uses chromadb.EphemeralClient() (in-memory, no file I/O) so tests run offline and
leave no artifacts.

IMPORTANT: chromadb 1.5.x EphemeralClient instances share a single in-memory store —
they are not isolated per instance. Each setup_method creates a client with
`Settings(allow_reset=True)` and calls `reset()` to wipe all collections before every
test, ensuring no residual state leaks between tests.

Key contracts verified:
  - All Chroma metadata values are primitive types (str/int/float/bool).
  - Span tuples are encoded as "start:end" strings.
  - Citation metadata fields (source_name, url, tier, published_date, title,
    identity_key) all survive the round-trip.
  - Cosine distance space is set at collection creation (via collection.configuration).
  - Upsert is idempotent: same chunk_id twice → count stays 1.
  - L2 stale deletion: new chunks with the same identity_key evict old chunks before
    the new ones are written.
  - Collection name encodes the embedder's model_name slug.
  - ModelMismatchError is raised when the stored model slug differs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("chromadb", reason="chromadb not installed")

import chromadb

from extraction.chroma_store import ChromaVectorStore, ModelMismatchError, _to_metadata
from extraction.chunk import Chunk, build_chunk
from tests.extraction.fixtures import make_document

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "fake-384d"
_DIM = 384
# Settings enabling reset() so setup_method can wipe shared in-memory state.
_SETTINGS = chromadb.Settings(allow_reset=True)


def _fresh_client() -> chromadb.ClientAPI:
    client = chromadb.EphemeralClient(settings=_SETTINGS)
    client.reset()
    return client


def _fake_vector() -> list[float]:
    return [0.1] * _DIM


def _make_chunk(ordinal: int = 0, **doc_overrides) -> Chunk:
    doc = make_document(**doc_overrides)
    body_slice = doc.body[:20] if doc.body else "placeholder text xxxx"
    return build_chunk(
        doc, ordinal, body_slice, (0, 20), (0, 10), chunked_at="2026-01-01T00:00:00Z"
    )


def _make_store(client: chromadb.ClientAPI) -> ChromaVectorStore:
    return ChromaVectorStore(client, _MODEL)


# ---------------------------------------------------------------------------
# _to_metadata: all values must be primitive
# ---------------------------------------------------------------------------


class TestToMetadata:
    def test_all_values_are_primitive(self):
        chunk = _make_chunk()
        meta = _to_metadata(chunk)
        for key, val in meta.items():
            assert isinstance(val, (str, int, float, bool)), (
                f"metadata['{key}'] = {val!r} is not a primitive type"
            )

    def test_full_span_encoded_as_string(self):
        assert _to_metadata(_make_chunk())["full_span"] == "0:20"

    def test_highlight_span_encoded_as_string(self):
        assert _to_metadata(_make_chunk())["highlight_span"] == "0:10"

    def test_tier_is_int(self):
        assert isinstance(_to_metadata(_make_chunk())["tier"], int)

    def test_all_citation_keys_present(self):
        required = {
            "chunk_id", "document_id", "identity_key", "ordinal", "source_name", "url",
            "tier", "published_date", "title", "chunked_at",
            "full_span", "highlight_span",
        }
        assert required.issubset(_to_metadata(_make_chunk()).keys())

    def test_identity_key_preserved(self):
        doc = make_document(identity_key="reuters::abc")
        chunk = build_chunk(doc, 0, "x", (0, 1), (0, 1), chunked_at="t")
        assert _to_metadata(chunk)["identity_key"] == "reuters::abc"

    def test_published_epoch_derived_from_published_date(self):
        # retrieval/search.py's build_where_clause filters on metadata["published_epoch"]
        # (an int, for Chroma's $gte) and RetrievedChunk.published_epoch reads it back —
        # published_date alone (an ISO string) can't serve either purpose.
        doc = make_document(published_date=datetime(2026, 1, 1, tzinfo=timezone.utc))
        chunk = build_chunk(doc, 0, "x", (0, 1), (0, 1), chunked_at="t")
        meta = _to_metadata(chunk)
        assert isinstance(meta["published_epoch"], int)
        assert meta["published_epoch"] == int(
            datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
        )

    def test_ordinal_preserved(self):
        # retrieval/search.py reads metadata["ordinal"] to populate RetrievedChunk.ordinal,
        # which output.py's citation_fields_present treats as a required field — omitting
        # it here would make every real (non-fake) retrieval result flag as degraded.
        assert _to_metadata(_make_chunk(ordinal=3))["ordinal"] == 3

    def test_section_label_present_when_set(self):
        doc = make_document()
        chunk = build_chunk(
            doc, 0, "x", (0, 1), (0, 1), chunked_at="t", section_label="RISK FACTORS"
        )
        assert _to_metadata(chunk)["section_label"] == "RISK FACTORS"

    def test_section_label_key_omitted_when_none(self):
        # Chroma metadata values must be primitives — None is not one, so a paragraph/fixed
        # chunk (section_label=None by design) must omit the key rather than write None.
        assert "section_label" not in _to_metadata(_make_chunk())


# ---------------------------------------------------------------------------
# ChromaVectorStore: collection creation and naming
# ---------------------------------------------------------------------------


class TestCollectionCreation:
    def setup_method(self):
        self.client = _fresh_client()

    def test_collection_name_encodes_model(self):
        store = _make_store(self.client)
        assert store.collection_name == f"mosaic_{_MODEL}"

    def test_collection_uses_cosine_distance(self):
        store = _make_store(self.client)
        store.upsert([_make_chunk()], [_fake_vector()])
        col = self.client.get_collection(store.collection_name)
        # Cosine distance is stored in collection.configuration, not in metadata.
        assert col.configuration["hnsw"]["space"] == "cosine"

    def test_model_mismatch_raises(self):
        # Create a collection, then corrupt its stored embedder_model to simulate
        # a collection built by a different model — the next store accessing it should raise.
        store_a = ChromaVectorStore(self.client, "model-a")
        store_a.upsert([_make_chunk()], [_fake_vector()])
        col = self.client.get_collection("mosaic_model-a")
        col.modify(metadata={"embedder_model": "other-model"})

        store_bad = ChromaVectorStore.__new__(ChromaVectorStore)
        store_bad._client = self.client
        store_bad._model_name = "model-a"
        store_bad._collection = None
        with pytest.raises(ModelMismatchError):
            store_bad._get_or_create_collection()


# ---------------------------------------------------------------------------
# ChromaVectorStore: upsert contract
# ---------------------------------------------------------------------------


class TestUpsert:
    def setup_method(self):
        self.client = _fresh_client()
        self.store = _make_store(self.client)

    def _col(self):
        return self.client.get_collection(self.store.collection_name)

    def test_upsert_stores_vector_and_document(self):
        chunk = _make_chunk()
        self.store.upsert([chunk], [_fake_vector()])
        result = self._col().get(ids=[chunk.chunk_id], include=["documents"])
        assert result["documents"][0] == chunk.text

    def test_upsert_stores_all_citation_metadata(self):
        doc = make_document(
            source_name="Reuters",
            url="https://reuters.com/a",
            tier=0,
            published_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            title="Big News",
            identity_key="reuters::big-news",
        )
        chunk = build_chunk(
            doc, 0, doc.body[:20], (0, 20), (0, 10), chunked_at="2026-01-01T00:00:01Z"
        )
        self.store.upsert([chunk], [_fake_vector()])
        result = self._col().get(ids=[chunk.chunk_id], include=["metadatas"])
        meta = result["metadatas"][0]
        assert meta["source_name"] == "Reuters"
        assert meta["url"] == "https://reuters.com/a"
        assert meta["tier"] == 0
        assert meta["published_date"] == "2026-01-01T00:00:00+00:00"
        assert meta["title"] == "Big News"
        assert meta["identity_key"] == "reuters::big-news"
        assert meta["ordinal"] == 0

    def test_upsert_empty_batch_is_noop(self):
        self.store.upsert([], [])
        collections = self.client.list_collections()
        assert not any(c.name == self.store.collection_name for c in collections)

    def test_upsert_is_idempotent(self):
        chunk = _make_chunk()
        self.store.upsert([chunk], [_fake_vector()])
        self.store.upsert([chunk], [_fake_vector()])
        assert self._col().count() == 1

    def test_l2_stale_deletion_removes_old_chunks(self):
        # First version of the document: 2 chunks.
        doc_v1 = make_document(id="doc-v1", identity_key="src::article-1")
        chunk_v1_a = build_chunk(doc_v1, 0, "old text a", (0, 10), (0, 5), chunked_at="t1")
        chunk_v1_b = build_chunk(doc_v1, 1, "old text b", (10, 20), (10, 15), chunked_at="t1")
        self.store.upsert([chunk_v1_a, chunk_v1_b], [_fake_vector(), _fake_vector()])
        assert self._col().count() == 2

        # Second version: same identity_key, new document_id, 1 new chunk.
        doc_v2 = make_document(id="doc-v2", identity_key="src::article-1")
        chunk_v2 = build_chunk(doc_v2, 0, "new text", (0, 8), (0, 4), chunked_at="t2")
        self.store.upsert([chunk_v2], [_fake_vector()])

        # Old 2 chunks evicted; only the new chunk remains.
        assert self._col().count() == 1
        result = self._col().get(include=["metadatas"])
        assert result["metadatas"][0]["document_id"] == "doc-v2"

    def test_multiple_documents_coexist(self):
        # Chunks from two different identity_keys must not evict each other.
        doc_a = make_document(id="doc-a", identity_key="src::a")
        doc_b = make_document(id="doc-b", identity_key="src::b")
        chunk_a = build_chunk(doc_a, 0, "text a", (0, 6), (0, 3), chunked_at="t")
        chunk_b = build_chunk(doc_b, 0, "text b", (0, 6), (0, 3), chunked_at="t")
        self.store.upsert([chunk_a], [_fake_vector()])
        self.store.upsert([chunk_b], [_fake_vector()])
        assert self._col().count() == 2
