"""Tests for ingestion/pipeline/dedup.py.

Contract under test:
  classify(doc: Document, seen_store: SeenStore, embedding: list[float] | None = None)
      -> DedupResult

  DedupResult enum: NEW | L1_DUPLICATE | L2_UPDATE | L3_NEAR_DUPLICATE

  L1 — content_hash already in seen_store → discard (byte-for-byte duplicate)
       L1 takes priority over L2: same hash means same content regardless of identity_key.

  L2 — identity_key in store with a *different* content_hash → article was updated
       Action: ingest the new version; do NOT discard.

  L3 — no identity_key match, but embedding similarity exceeds threshold → same story,
       different outlet.
       Action: ingest BOTH. Never discard L3 — cross-outlet corroboration must be preserved.

  NEW — no match at any level → ingest normally.

Note: MockSeenStore below stands in for ingestion/storage/seen_store.py until that is implemented.
"""

from ingestion.pipeline.dedup import DedupResult, classify
from tests.conftest import make_document

# ---------------------------------------------------------------------------
# Test double for SeenStore (real impl lives in ingestion/storage/seen_store.py)
# ---------------------------------------------------------------------------


class MockSeenStore:
    """Minimal in-memory SeenStore for testing dedup logic in isolation."""

    def __init__(self):
        self._hashes: set = set()
        self._identity_keys: dict = {}  # identity_key -> content_hash
        self._embeddings: list = []  # list of (doc_id, embedding_vector)

    def contains_hash(self, content_hash: str) -> bool:
        return content_hash in self._hashes

    def get_hash(self, identity_key: str) -> str | None:
        return self._identity_keys.get(identity_key)

    def get_embeddings(self) -> list:
        """Returns list of (doc_id, embedding_vector) for L3 comparison."""
        return self._embeddings

    def add(self, doc, embedding: list | None = None) -> None:
        self._hashes.add(doc.content_hash)
        self._identity_keys[doc.identity_key] = doc.content_hash
        if embedding is not None:
            self._embeddings.append((doc.id, embedding))


# ---------------------------------------------------------------------------
# L1 — Exact duplicate (content hash match)
# ---------------------------------------------------------------------------


class TestL1ExactDuplicate:
    def test_same_content_hash_is_l1(self):
        store = MockSeenStore()
        original = make_document(
            content_hash="hash_abc", identity_key="Reuters::article-1"
        )
        store.add(original)

        duplicate = make_document(
            content_hash="hash_abc", identity_key="Reuters::article-99"
        )
        assert classify(duplicate, store) == DedupResult.L1_DUPLICATE

    def test_different_identity_key_still_l1_if_hash_matches(self):
        store = MockSeenStore()
        store._hashes.add("shared_hash")
        store._identity_keys["Bloomberg::article-5"] = "shared_hash"

        incoming = make_document(
            content_hash="shared_hash", identity_key="Reuters::article-1"
        )
        assert classify(incoming, store) == DedupResult.L1_DUPLICATE

    def test_l1_takes_priority_over_l2(self):
        # Same identity key AND same content hash → L1 wins (content unchanged, not an update)
        store = MockSeenStore()
        original = make_document(
            content_hash="hash_xyz", identity_key="Reuters::article-1"
        )
        store.add(original)

        same_everything = make_document(
            content_hash="hash_xyz", identity_key="Reuters::article-1"
        )
        assert classify(same_everything, store) == DedupResult.L1_DUPLICATE

    def test_unseen_hash_is_not_l1(self):
        store = MockSeenStore()
        store._hashes.add("old_hash")

        new_doc = make_document(content_hash="brand_new_hash")
        assert classify(new_doc, store) != DedupResult.L1_DUPLICATE


# ---------------------------------------------------------------------------
# L2 — Same article, updated (identity key match, different content hash)
# ---------------------------------------------------------------------------


class TestL2UpdatedArticle:
    def test_same_identity_key_different_hash_is_l2(self):
        store = MockSeenStore()
        original = make_document(
            content_hash="hash_v1", identity_key="Reuters::article-1"
        )
        store.add(original)

        updated = make_document(
            content_hash="hash_v2",
            identity_key="Reuters::article-1",
            title="Fed cuts rates by 25 basis points (Updated)",
        )
        assert classify(updated, store) == DedupResult.L2_UPDATE

    def test_different_identity_key_is_not_l2(self):
        store = MockSeenStore()
        original = make_document(
            content_hash="hash_v1", identity_key="Reuters::article-1"
        )
        store.add(original)

        unrelated = make_document(
            content_hash="hash_v2", identity_key="Reuters::article-2"
        )
        assert classify(unrelated, store) != DedupResult.L2_UPDATE

    def test_l2_result_means_article_was_updated_not_duplicated(self):
        store = MockSeenStore()
        original = make_document(content_hash="old", identity_key="Reuters::article-1")
        store.add(original)

        updated = make_document(content_hash="new", identity_key="Reuters::article-1")
        result = classify(updated, store)
        # L2 ≠ L1 — caller must NOT discard; it must store the new version
        assert result == DedupResult.L2_UPDATE
        assert result != DedupResult.L1_DUPLICATE


# ---------------------------------------------------------------------------
# L3 — Near-duplicate / same story, different outlet (embedding similarity)
# ---------------------------------------------------------------------------


class TestL3NearDuplicate:
    def test_high_cosine_similarity_is_l3(self):
        store = MockSeenStore()
        existing = make_document(
            content_hash="hash_bloomberg",
            identity_key="Bloomberg::article-1",
        )
        near_identical_embedding = [1.0, 0.0, 0.0]
        store.add(existing, embedding=near_identical_embedding)

        incoming = make_document(
            content_hash="hash_reuters",
            identity_key="Reuters::article-2",
            source_name="Reuters",
        )
        result = classify(incoming, store, embedding=[1.0, 0.0, 0.0])
        assert result == DedupResult.L3_NEAR_DUPLICATE

    def test_low_similarity_is_not_l3(self):
        store = MockSeenStore()
        existing = make_document(
            content_hash="hash_bloomberg",
            identity_key="Bloomberg::article-1",
        )
        store.add(existing, embedding=[1.0, 0.0, 0.0])

        incoming = make_document(
            content_hash="hash_reuters",
            identity_key="Reuters::article-99",
        )
        # Orthogonal embedding → no semantic similarity
        result = classify(incoming, store, embedding=[0.0, 1.0, 0.0])
        assert result != DedupResult.L3_NEAR_DUPLICATE

    def test_l3_is_not_l1_or_l2(self):
        # L3 is its own distinct result — not conflated with exact or updated dedup
        store = MockSeenStore()
        store.add(
            make_document(content_hash="hash_A", identity_key="Bloomberg::article-1"),
            embedding=[1.0, 0.0, 0.0],
        )

        incoming = make_document(
            content_hash="hash_B", identity_key="Reuters::article-2"
        )
        result = classify(incoming, store, embedding=[1.0, 0.0, 0.0])
        assert result == DedupResult.L3_NEAR_DUPLICATE
        assert result != DedupResult.L1_DUPLICATE
        assert result != DedupResult.L2_UPDATE

    def test_l3_preserves_cross_outlet_corroboration(self):
        # L3 MUST NOT cause a discard — both docs must be ingested for trust assessment
        store = MockSeenStore()
        store.add(
            make_document(
                source_name="Bloomberg",
                content_hash="hash_bloomberg",
                identity_key="Bloomberg::article-1",
            ),
            embedding=[1.0, 0.0, 0.0],
        )

        incoming = make_document(
            source_name="Reuters",
            content_hash="hash_reuters",
            identity_key="Reuters::article-2",
        )
        result = classify(incoming, store, embedding=[1.0, 0.0, 0.0])
        # The caller must ingest this document — L3 is informational, not a discard signal
        assert result == DedupResult.L3_NEAR_DUPLICATE


# ---------------------------------------------------------------------------
# NEW — No match at any level
# ---------------------------------------------------------------------------


class TestNewDocument:
    def test_empty_store_always_new(self):
        store = MockSeenStore()
        doc = make_document()
        assert classify(doc, store) == DedupResult.NEW

    def test_unseen_hash_and_unseen_identity_key_is_new(self):
        store = MockSeenStore()
        store.add(
            make_document(content_hash="old_hash", identity_key="Reuters::old-article")
        )

        incoming = make_document(
            content_hash="brand_new_hash",
            identity_key="Reuters::new-article",
        )
        assert classify(incoming, store) == DedupResult.NEW

    def test_new_with_no_similar_embeddings(self):
        store = MockSeenStore()
        store.add(
            make_document(content_hash="hash_A", identity_key="Reuters::article-1"),
            embedding=[1.0, 0.0, 0.0],
        )

        incoming = make_document(
            content_hash="hash_B",
            identity_key="Reuters::article-2",
        )
        result = classify(incoming, store, embedding=[0.0, 0.0, 1.0])
        assert result == DedupResult.NEW
