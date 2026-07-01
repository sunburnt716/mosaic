"""Tests for ingestion/storage/poll_state.py.

Covers PollState construction and its helper methods (validators_fresh,
conditional_headers). Store-level round-trip tests live in test_storage.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ingestion.storage.poll_state import PollState, PollStateStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# PollState construction
# ---------------------------------------------------------------------------


class TestPollStateConstruction:
    def test_source_name_stored(self):
        state = PollState("reuters-rss", None, None, None, None)
        assert state.source_name == "reuters-rss"

    def test_none_last_polled_at_accepted(self):
        state = PollState("src", None, None, None, None)
        assert state.last_polled_at is None

    def test_utc_aware_datetime_stored(self):
        ts = utc(2024, 1, 15, 10, 30)
        state = PollState("src", ts, None, None, None)
        assert state.last_polled_at == ts

    def test_etag_and_last_modified_stored(self):
        state = PollState("src", None, '"abc123"', "Mon, 15 Jan 2024 00:00:00 GMT", None)
        assert state.etag == '"abc123"'
        assert state.last_modified == "Mon, 15 Jan 2024 00:00:00 GMT"

    def test_etag_cached_at_stored(self):
        ts = utc(2024, 1, 15)
        state = PollState("src", None, '"abc"', None, ts)
        assert state.etag_cached_at == ts

    def test_all_none_fields_accepted(self):
        state = PollState("src", None, None, None, None)
        assert state.etag is None
        assert state.last_modified is None
        assert state.etag_cached_at is None


# ---------------------------------------------------------------------------
# validators_fresh
# ---------------------------------------------------------------------------


class TestValidatorsFresh:
    def test_false_when_etag_cached_at_is_none(self):
        state = PollState("src", None, '"abc"', None, None)
        assert state.validators_fresh() is False

    def test_true_when_cached_recently(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        state = PollState("src", None, '"abc"', None, recent)
        assert state.validators_fresh() is True

    def test_false_when_cached_over_24h_ago(self):
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        state = PollState("src", None, '"abc"', None, old)
        assert state.validators_fresh() is False

    def test_false_when_no_etag_and_no_cached_at(self):
        state = PollState("src", None, None, None, None)
        assert state.validators_fresh() is False


# ---------------------------------------------------------------------------
# conditional_headers
# ---------------------------------------------------------------------------


class TestConditionalHeaders:
    def test_empty_when_stale(self):
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        state = PollState("src", None, '"abc"', "some-date", old)
        assert state.conditional_headers() == {}

    def test_empty_when_no_validators(self):
        state = PollState("src", None, None, None, None)
        assert state.conditional_headers() == {}

    def test_includes_if_none_match_when_fresh(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        state = PollState("src", None, '"abc"', None, recent)
        headers = state.conditional_headers()
        assert headers.get("If-None-Match") == '"abc"'

    def test_includes_if_modified_since_when_fresh(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        state = PollState("src", None, None, "Mon, 15 Jan 2024 00:00:00 GMT", recent)
        headers = state.conditional_headers()
        assert headers.get("If-Modified-Since") == "Mon, 15 Jan 2024 00:00:00 GMT"

    def test_omits_if_none_match_when_etag_is_none(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        state = PollState("src", None, None, "some-date", recent)
        assert "If-None-Match" not in state.conditional_headers()


# ---------------------------------------------------------------------------
# PollStateStore — basic smoke tests (full suite in test_storage.py)
# ---------------------------------------------------------------------------


class TestPollStateStoreSmokeTests:
    @pytest.fixture
    def store(self):
        s = PollStateStore(":memory:")
        yield s
        s.close()

    def test_get_blank_state_for_unknown_source(self, store):
        state = store.get("unknown")
        assert isinstance(state, PollState)
        assert state.last_polled_at is None

    def test_touch_records_poll_time(self, store):
        store.touch("reuters-rss")
        assert store.get("reuters-rss").last_polled_at is not None

    def test_update_stores_etag(self, store):
        store.update("reuters-rss", etag='"v1"', last_modified=None)
        assert store.get("reuters-rss").etag == '"v1"'

    def test_multiple_sources_are_independent(self, store):
        store.update("src-a", etag='"a"', last_modified=None)
        store.update("src-b", etag='"b"', last_modified=None)
        assert store.get("src-a").etag == '"a"'
        assert store.get("src-b").etag == '"b"'
