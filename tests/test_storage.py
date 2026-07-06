"""Tests for ingestion/storage/{seen_store,raw_store}.py.

PollStateStore/PollState are covered separately in test_poll_state.py (their own,
richer API — this file only covers the two dedup-adjacent stores). All tests use
in-memory SQLite (:memory:) so they are fast and leave no files on disk.
"""

import pytest

from ingestion.storage.raw_store import RawStore
from ingestion.storage.seen_store import SeenStore
from tests.conftest import make_document

# ---------------------------------------------------------------------------
# SeenStore
# ---------------------------------------------------------------------------


class TestSeenStore:
    @pytest.fixture
    def store(self):
        s = SeenStore(":memory:")
        yield s
        s.close()

    def test_contains_hash_false_when_empty(self, store):
        assert store.contains_hash("abc123") is False

    def test_set_and_contains_hash(self, store):
        store.set_hash("Reuters::article-1", "hash_abc")
        assert store.contains_hash("hash_abc") is True

    def test_contains_hash_false_for_different_hash(self, store):
        store.set_hash("Reuters::article-1", "hash_abc")
        assert store.contains_hash("other_hash") is False

    def test_get_hash_returns_none_when_absent(self, store):
        assert store.get_hash("Reuters::unknown") is None

    def test_get_hash_returns_stored_value(self, store):
        store.set_hash("Reuters::article-1", "hash_abc")
        assert store.get_hash("Reuters::article-1") == "hash_abc"

    def test_set_hash_overwrites_existing(self, store):
        store.set_hash("Reuters::article-1", "hash_v1")
        store.set_hash("Reuters::article-1", "hash_v2")
        assert store.get_hash("Reuters::article-1") == "hash_v2"

    def test_contains_hash_true_after_overwrite(self, store):
        store.set_hash("Reuters::article-1", "hash_v1")
        store.set_hash("Reuters::article-1", "hash_v2")
        assert store.contains_hash("hash_v2") is True

    def test_old_hash_removed_after_overwrite(self, store):
        store.set_hash("Reuters::article-1", "hash_v1")
        store.set_hash("Reuters::article-1", "hash_v2")
        assert store.contains_hash("hash_v1") is False

    def test_multiple_identity_keys_independent(self, store):
        store.set_hash("Reuters::a", "hash_a")
        store.set_hash("Reuters::b", "hash_b")
        assert store.get_hash("Reuters::a") == "hash_a"
        assert store.get_hash("Reuters::b") == "hash_b"


# ---------------------------------------------------------------------------
# RawStore
# ---------------------------------------------------------------------------


class TestRawStore:
    @pytest.fixture
    def store(self):
        s = RawStore(":memory:")
        yield s
        s.close()

    def test_get_document_returns_none_when_absent(self, store):
        assert store.get_document("nonexistent") is None

    def test_get_raw_returns_none_when_absent(self, store):
        assert store.get_raw("nonexistent") is None

    def test_save_and_get_raw(self, store):
        payload = {"key": "value", "nested": {"a": 1}}
        store.save_raw("doc-1", payload)
        assert store.get_raw("doc-1") == payload

    def test_save_raw_is_idempotent(self, store):
        store.save_raw("doc-1", {"v": 1})
        store.save_raw("doc-1", {"v": 2})  # second write ignored (INSERT OR IGNORE)
        assert store.get_raw("doc-1") == {"v": 1}

    def test_save_and_get_document(self, store):
        doc = make_document()
        store.save_document(doc)
        retrieved = store.get_document(doc.id)
        assert retrieved is not None
        assert retrieved.id == doc.id
        assert retrieved.title == doc.title
        assert retrieved.tier == doc.tier

    def test_get_document_preserves_utc_datetime(self, store):
        doc = make_document()
        store.save_document(doc)
        retrieved = store.get_document(doc.id)
        assert retrieved.published_date.tzinfo is not None

    def test_save_document_overwrites_on_l2_update(self, store):
        doc_v1 = make_document(title="Original title")
        doc_v2 = make_document(title="Updated title")  # same id by default
        store.save_document(doc_v1)
        store.save_document(doc_v2)
        retrieved = store.get_document(doc_v1.id)
        assert retrieved.title == "Updated title"

    def test_raw_and_document_stored_independently(self, store):
        doc = make_document()
        store.save_raw(doc.id, {"raw": True})
        store.save_document(doc)
        assert store.get_raw(doc.id) == {"raw": True}
        assert store.get_document(doc.id).id == doc.id
