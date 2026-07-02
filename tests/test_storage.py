"""Tests for ingestion/storage/{seen_store,raw_store,poll_state}.py.

All tests use in-memory SQLite (:memory:) so they are fast and leave no files on disk.
"""

from datetime import datetime, timedelta, timezone

import pytest

from ingestion.storage.poll_state import PollState, PollStateStore
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


# ---------------------------------------------------------------------------
# PollStateStore
# ---------------------------------------------------------------------------


class TestPollStateStore:
    @pytest.fixture
    def store(self):
        s = PollStateStore(":memory:")
        yield s
        s.close()

    def test_get_returns_blank_state_for_unknown_source(self, store):
        state = store.get("unknown-source")
        assert isinstance(state, PollState)
        assert state.last_polled_at is None
        assert state.etag is None
        assert state.last_modified is None

    def test_touch_sets_last_polled_at(self, store):
        store.touch("reuters-rss")
        state = store.get("reuters-rss")
        assert state.last_polled_at is not None

    def test_touch_does_not_overwrite_validators(self, store):
        store.update(
            "reuters-rss", etag='"abc"', last_modified="Mon, 15 Jan 2024 14:30:00 GMT"
        )
        store.touch("reuters-rss")
        state = store.get("reuters-rss")
        assert state.etag == '"abc"'

    def test_update_stores_validators(self, store):
        store.update(
            "reuters-rss",
            etag='"abc123"',
            last_modified="Mon, 15 Jan 2024 14:30:00 GMT",
        )
        state = store.get("reuters-rss")
        assert state.etag == '"abc123"'
        assert state.last_modified == "Mon, 15 Jan 2024 14:30:00 GMT"
        assert state.etag_cached_at is not None

    def test_update_resets_etag_cached_at(self, store):
        store.update("reuters-rss", etag='"v1"', last_modified=None)
        t1 = store.get("reuters-rss").etag_cached_at
        store.update("reuters-rss", etag='"v2"', last_modified=None)
        t2 = store.get("reuters-rss").etag_cached_at
        assert t2 >= t1

    def test_validators_fresh_within_24h(self, store):
        store.update("reuters-rss", etag='"abc"', last_modified=None)
        state = store.get("reuters-rss")
        assert state.validators_fresh() is True

    def test_validators_not_fresh_after_24h(self):
        state = PollState(
            source_name="reuters-rss",
            last_polled_at=None,
            etag='"abc"',
            last_modified=None,
            etag_cached_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        assert state.validators_fresh() is False

    def test_validators_not_fresh_when_never_set(self):
        state = PollState("source", None, None, None, None)
        assert state.validators_fresh() is False

    def test_conditional_headers_empty_when_stale(self):
        state = PollState(
            source_name="reuters-rss",
            last_polled_at=None,
            etag='"abc"',
            last_modified="Mon, 15 Jan 2024 14:30:00 GMT",
            etag_cached_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        assert state.conditional_headers() == {}

    def test_conditional_headers_set_when_fresh(self, store):
        store.update(
            "reuters-rss", etag='"abc"', last_modified="Mon, 15 Jan 2024 14:30:00 GMT"
        )
        state = store.get("reuters-rss")
        headers = state.conditional_headers()
        assert headers.get("If-None-Match") == '"abc"'
        assert "If-Modified-Since" in headers

    def test_update_with_none_validators_clears_them(self, store):
        store.update("reuters-rss", etag='"v1"', last_modified="some-date")
        store.update("reuters-rss", etag=None, last_modified=None)
        state = store.get("reuters-rss")
        assert state.etag is None
