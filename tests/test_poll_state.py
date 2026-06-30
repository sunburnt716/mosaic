"""
Tests for ingestion/storage/poll_state.py.

Covers:
  - PollState construction: valid UTC datetime, naive datetime rejected,
    None last_polled_at accepted.
  - PollStateStore: get on missing source, round-trip set/get, update,
    persistence across instances, corrupt file degrades gracefully,
    atomic write (no temp file left behind).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ingestion.storage.poll_state import PollState, PollStateStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Construct a UTC-aware datetime for test fixtures."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# PollState construction
# ---------------------------------------------------------------------------


class TestPollState:
    def test_none_last_polled_at_accepted(self):
        state = PollState(last_polled_at=None, etag=None, last_modified=None)
        assert state.last_polled_at is None

    def test_utc_aware_datetime_accepted(self):
        ts = utc(2024, 1, 15, 10, 30)
        state = PollState(last_polled_at=ts, etag=None, last_modified=None)
        assert state.last_polled_at == ts

    def test_naive_datetime_raises(self):
        naive = datetime(2024, 1, 15, 10, 30)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            PollState(last_polled_at=naive, etag=None, last_modified=None)

    def test_etag_and_last_modified_stored(self):
        ts = utc(2024, 1, 15)
        state = PollState(
            last_polled_at=ts, etag='"abc123"', last_modified="Mon, 15 Jan 2024 00:00:00 GMT"
        )
        assert state.etag == '"abc123"'
        assert state.last_modified == "Mon, 15 Jan 2024 00:00:00 GMT"

    def test_is_frozen(self):
        state = PollState(last_polled_at=None, etag=None, last_modified=None)
        with pytest.raises((AttributeError, TypeError)):
            state.etag = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PollStateStore — basic reads
# ---------------------------------------------------------------------------


class TestPollStateStoreReads:
    def test_get_returns_none_before_any_write(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "poll_state.json")
        assert store.get("some-source") is None

    def test_get_returns_none_for_unknown_source(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "poll_state.json")
        store.set(
            "source-a",
            PollState(last_polled_at=utc(2024, 1, 1), etag=None, last_modified=None),
        )
        assert store.get("source-b") is None

    def test_missing_state_file_returns_none(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "does-not-exist.json")
        assert store.get("any-source") is None


# ---------------------------------------------------------------------------
# PollStateStore — round-trip
# ---------------------------------------------------------------------------


class TestPollStateStoreRoundTrip:
    def test_set_and_get_full_state(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "poll_state.json")
        ts = utc(2024, 3, 12, 14, 55)
        original = PollState(
            last_polled_at=ts,
            etag='"etag-value"',
            last_modified="Tue, 12 Mar 2024 14:55:00 GMT",
        )
        store.set("reuters-rss", original)
        retrieved = store.get("reuters-rss")
        assert retrieved is not None
        assert retrieved.last_polled_at == ts
        assert retrieved.etag == '"etag-value"'
        assert retrieved.last_modified == "Tue, 12 Mar 2024 14:55:00 GMT"

    def test_set_and_get_none_fields(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "poll_state.json")
        original = PollState(last_polled_at=None, etag=None, last_modified=None)
        store.set("source", original)
        retrieved = store.get("source")
        assert retrieved is not None
        assert retrieved.last_polled_at is None
        assert retrieved.etag is None

    def test_last_polled_at_timezone_preserved(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "poll_state.json")
        ts = utc(2024, 6, 1, 12, 0)
        store.set("source", PollState(last_polled_at=ts, etag=None, last_modified=None))
        retrieved = store.get("source")
        assert retrieved.last_polled_at.tzinfo is not None
        assert retrieved.last_polled_at == ts

    def test_set_updates_existing_entry(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "poll_state.json")
        ts1 = utc(2024, 1, 1)
        ts2 = utc(2024, 1, 2)
        store.set("source", PollState(last_polled_at=ts1, etag=None, last_modified=None))
        store.set("source", PollState(last_polled_at=ts2, etag='"new"', last_modified=None))
        retrieved = store.get("source")
        assert retrieved.last_polled_at == ts2
        assert retrieved.etag == '"new"'

    def test_multiple_sources_independent(self, tmp_path: Path):
        store = PollStateStore(tmp_path / "poll_state.json")
        ts_a = utc(2024, 1, 1)
        ts_b = utc(2024, 2, 1)
        store.set("source-a", PollState(last_polled_at=ts_a, etag="a", last_modified=None))
        store.set("source-b", PollState(last_polled_at=ts_b, etag="b", last_modified=None))
        a = store.get("source-a")
        b = store.get("source-b")
        assert a.last_polled_at == ts_a
        assert b.last_polled_at == ts_b
        assert a.etag == "a"
        assert b.etag == "b"


# ---------------------------------------------------------------------------
# PollStateStore — persistence across instances
# ---------------------------------------------------------------------------


class TestPollStateStorePersistence:
    def test_data_survives_new_store_instance(self, tmp_path: Path):
        path = tmp_path / "poll_state.json"
        ts = utc(2024, 5, 20, 8, 0)
        # Write with first instance.
        store1 = PollStateStore(path)
        store1.set("source", PollState(last_polled_at=ts, etag=None, last_modified=None))
        # Read with a second instance pointing to the same file.
        store2 = PollStateStore(path)
        retrieved = store2.get("source")
        assert retrieved is not None
        assert retrieved.last_polled_at == ts

    def test_file_is_valid_json(self, tmp_path: Path):
        path = tmp_path / "poll_state.json"
        store = PollStateStore(path)
        store.set("s", PollState(last_polled_at=utc(2024, 1, 1), etag=None, last_modified=None))
        raw = json.loads(path.read_text())
        assert "s" in raw
        assert "last_polled_at" in raw["s"]


# ---------------------------------------------------------------------------
# PollStateStore — resilience
# ---------------------------------------------------------------------------


class TestPollStateStoreResilience:
    def test_corrupt_json_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "poll_state.json"
        path.write_text("{ not valid json }", encoding="utf-8")
        store = PollStateStore(path)
        # Should degrade gracefully — corrupt file means no state, not a crash.
        assert store.get("any-source") is None

    def test_corrupt_file_does_not_prevent_new_writes(self, tmp_path: Path):
        path = tmp_path / "poll_state.json"
        path.write_text("corrupt", encoding="utf-8")
        store = PollStateStore(path)
        ts = utc(2024, 1, 1)
        store.set("source", PollState(last_polled_at=ts, etag=None, last_modified=None))
        assert store.get("source").last_polled_at == ts

    def test_atomic_write_no_tmp_file_left(self, tmp_path: Path):
        path = tmp_path / "poll_state.json"
        store = PollStateStore(path)
        store.set("s", PollState(last_polled_at=utc(2024, 1, 1), etag=None, last_modified=None))
        tmp = path.with_suffix(".tmp")
        assert not tmp.exists(), "Temp file should be cleaned up after successful write"

    def test_parent_directory_created_if_missing(self, tmp_path: Path):
        path = tmp_path / "subdir" / "poll_state.json"
        store = PollStateStore(path)
        store.set("s", PollState(last_polled_at=utc(2024, 1, 1), etag=None, last_modified=None))
        assert path.exists()

    def test_naive_datetime_stored_by_legacy_data_coerced_to_utc(self, tmp_path: Path):
        # Simulate a state file written before timezone enforcement was added.
        path = tmp_path / "poll_state.json"
        path.write_text(
            json.dumps({
                "source": {
                    "last_polled_at": "2024-01-01T00:00:00",
                    "etag": None,
                    "last_modified": None,
                }
            }),
            encoding="utf-8",
        )
        store = PollStateStore(path)
        retrieved = store.get("source")
        # The deserializer should coerce naive to UTC rather than raising.
        assert retrieved is not None
        assert retrieved.last_polled_at.tzinfo is not None
