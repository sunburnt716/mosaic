"""Tests for ingestion/run.py.

Covers:
  - _parse_interval: valid formats, invalid formats.
  - _sources_due: never polled, recently polled, overdue, disabled sources.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ingestion.core.source_config import SourceConfig
from ingestion.run import _parse_interval, _sources_due
from ingestion.storage.poll_state import PollState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def make_source(
    name: str = "test-source",
    poll_interval: str = "5m",
    enabled: bool = True,
) -> SourceConfig:
    return SourceConfig(
        name=name,
        adapter="rss",
        tier=1,
        url="https://example.com/feed.xml",
        poll_interval=poll_interval,
        enabled=enabled,
    )


class FakePollStateStore:
    """In-memory PollStateStore stub — returns blank state for unknown sources."""

    def __init__(self, states: dict[str, PollState] | None = None) -> None:
        self._states: dict[str, PollState] = states or {}

    def get(self, source_name: str) -> PollState:
        return self._states.get(
            source_name, PollState(source_name, None, None, None, None)
        )


# ---------------------------------------------------------------------------
# _parse_interval
# ---------------------------------------------------------------------------


class TestParseInterval:
    def test_minutes_only(self):
        assert _parse_interval("5m") == timedelta(minutes=5)

    def test_hours_only(self):
        assert _parse_interval("1h") == timedelta(hours=1)

    def test_seconds_only(self):
        assert _parse_interval("30s") == timedelta(seconds=30)

    def test_whitespace_stripped(self):
        assert _parse_interval("  5m  ") == timedelta(minutes=5)

    def test_large_minutes(self):
        assert _parse_interval("90m") == timedelta(minutes=90)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError):
            _parse_interval("5d")

    def test_empty_unit_raises(self):
        with pytest.raises((ValueError, IndexError)):
            _parse_interval("")


# ---------------------------------------------------------------------------
# _sources_due
# ---------------------------------------------------------------------------


class TestSourcesDue:
    NOW = utc(2024, 6, 1, 12, 0, 0)

    def test_never_polled_is_always_due(self):
        source = make_source("a")
        store = FakePollStateStore()  # blank state → last_polled_at is None
        due = _sources_due([source], store, now=self.NOW)
        assert [s.name for s in due] == ["a"]

    def test_just_polled_is_not_due(self):
        source = make_source("a", poll_interval="5m")
        store = FakePollStateStore(
            {"a": PollState("a", self.NOW, None, None, None)}
        )
        due = _sources_due([source], store, now=self.NOW)
        assert due == []

    def test_one_second_before_interval_not_due(self):
        source = make_source("a", poll_interval="5m")
        last = self.NOW - timedelta(minutes=5) + timedelta(seconds=1)
        store = FakePollStateStore({"a": PollState("a", last, None, None, None)})
        due = _sources_due([source], store, now=self.NOW)
        assert due == []

    def test_exactly_at_boundary_is_due(self):
        source = make_source("a", poll_interval="5m")
        last = self.NOW - timedelta(minutes=5)
        store = FakePollStateStore({"a": PollState("a", last, None, None, None)})
        due = _sources_due([source], store, now=self.NOW)
        assert [s.name for s in due] == ["a"]

    def test_long_overdue_is_due(self):
        source = make_source("a", poll_interval="5m")
        last = self.NOW - timedelta(days=7)
        store = FakePollStateStore({"a": PollState("a", last, None, None, None)})
        due = _sources_due([source], store, now=self.NOW)
        assert [s.name for s in due] == ["a"]

    def test_disabled_source_never_returned(self):
        source = make_source("a", enabled=False)
        store = FakePollStateStore()
        due = _sources_due([source], store, now=self.NOW)
        assert due == []

    def test_disabled_and_overdue_still_skipped(self):
        source = make_source("a", enabled=False, poll_interval="5m")
        last = self.NOW - timedelta(days=365)
        store = FakePollStateStore({"a": PollState("a", last, None, None, None)})
        due = _sources_due([source], store, now=self.NOW)
        assert due == []

    def test_mixed_enabled_disabled(self):
        sources = [
            make_source("enabled-due"),
            make_source("disabled", enabled=False),
        ]
        overdue = PollState("enabled-due", self.NOW - timedelta(hours=1), None, None, None)
        recent = PollState("disabled", self.NOW - timedelta(seconds=1), None, None, None)
        store = FakePollStateStore({"enabled-due": overdue, "disabled": recent})
        due = _sources_due(sources, store, now=self.NOW)
        assert [s.name for s in due] == ["enabled-due"]

    def test_empty_configs_returns_empty(self):
        store = FakePollStateStore()
        assert _sources_due([], store, now=self.NOW) == []

    def test_all_never_polled_all_returned(self):
        sources = [make_source("a"), make_source("b"), make_source("c")]
        store = FakePollStateStore()
        due = _sources_due(sources, store, now=self.NOW)
        assert [s.name for s in due] == ["a", "b", "c"]

    def test_different_poll_intervals_respected(self):
        short = make_source("short", poll_interval="2m")
        long_src = make_source("long", poll_interval="10m")
        last = self.NOW - timedelta(minutes=3)
        store = FakePollStateStore(
            {
                "short": PollState("short", last, None, None, None),
                "long": PollState("long", last, None, None, None),
            }
        )
        due = _sources_due([short, long_src], store, now=self.NOW)
        assert [s.name for s in due] == ["short"]

    def test_source_with_no_state_entry_is_treated_as_never_polled(self):
        source = make_source("fresh")
        store = FakePollStateStore(
            {"other": PollState("other", self.NOW, None, None, None)}
        )
        due = _sources_due([source], store, now=self.NOW)
        assert [s.name for s in due] == ["fresh"]
