"""
Tests for ingestion/run.py.

Covers:
  - is_due: never polled, just polled, exact boundary, one second before boundary,
    long overdue, future last_polled_at (clock skew).
  - select_due_sources: all never-polled, none due, mixed, disabled source skipped,
    disabled-but-overdue still skipped.
  - tick: dispatches due sources, isolates failures (one crash doesn't stop the rest),
    all fail, no due sources.
  - run_forever: stop_event set before start exits without ticking, runs ticks then
    stops cleanly.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from ingestion.core.source_config import SourceConfig, load_sources
from ingestion.run import is_due, run_forever, select_due_sources, tick
from ingestion.storage.poll_state import PollState, PollStateStore


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def utc(*args: int) -> datetime:
    """Construct a UTC-aware datetime. Arguments: year, month, day[, hour, minute, second]."""
    return datetime(*args, tzinfo=timezone.utc)


def make_source(
    name: str = "test-source",
    poll_interval_minutes: int = 5,
    enabled: bool = True,
    **kwargs: Any,
) -> SourceConfig:
    """Construct a minimal SourceConfig for test use."""
    return SourceConfig(
        name=name,
        adapter="rss",
        tier=1,
        url="https://example.com/feed.xml",
        poll_interval=timedelta(minutes=poll_interval_minutes),
        doc_type="article",
        field_mappings={},
        auth={},
        enabled=enabled,
        params={},
        headers={},
    )


class FakePollStateStore:
    """In-memory PollStateStore for tests — no file I/O."""

    def __init__(self, states: dict[str, PollState | None] | None = None) -> None:
        self._states: dict[str, PollState | None] = states or {}

    def get(self, source_name: str) -> PollState | None:
        return self._states.get(source_name)

    def set(self, source_name: str, state: PollState) -> None:
        self._states[source_name] = state


class StubEngine:
    """Records which sources were dispatched and optionally raises for specific ones."""

    def __init__(self, raises_for: set[str] | None = None) -> None:
        self.processed: list[str] = []
        self._raises_for: set[str] = raises_for or set()

    def process_source(self, source: SourceConfig) -> None:
        self.processed.append(source.name)
        if source.name in self._raises_for:
            raise RuntimeError(f"Simulated failure for {source.name}")


# ---------------------------------------------------------------------------
# is_due
# ---------------------------------------------------------------------------


class TestIsDue:
    NOW = utc(2024, 6, 1, 12, 0, 0)
    INTERVAL = timedelta(minutes=5)

    def _source(self) -> SourceConfig:
        return make_source(poll_interval_minutes=5)

    def test_never_polled_is_always_due(self):
        assert is_due(self._source(), last_polled_at=None, now=self.NOW) is True

    def test_just_polled_is_not_due(self):
        # Polled at exactly NOW — elapsed = 0, less than 5 minutes.
        assert is_due(self._source(), last_polled_at=self.NOW, now=self.NOW) is False

    def test_one_second_before_interval_is_not_due(self):
        last = self.NOW - self.INTERVAL + timedelta(seconds=1)
        assert is_due(self._source(), last_polled_at=last, now=self.NOW) is False

    def test_exactly_at_interval_boundary_is_due(self):
        # elapsed == poll_interval exactly → due.
        last = self.NOW - self.INTERVAL
        assert is_due(self._source(), last_polled_at=last, now=self.NOW) is True

    def test_one_second_past_interval_is_due(self):
        last = self.NOW - self.INTERVAL - timedelta(seconds=1)
        assert is_due(self._source(), last_polled_at=last, now=self.NOW) is True

    def test_long_overdue_is_due(self):
        last = self.NOW - timedelta(days=7)
        assert is_due(self._source(), last_polled_at=last, now=self.NOW) is True

    def test_future_last_polled_is_not_due(self):
        # Clock skew: last_polled_at is in the future relative to now.
        # elapsed is negative → not due. The function stays correct under skew.
        future = self.NOW + timedelta(minutes=10)
        assert is_due(self._source(), last_polled_at=future, now=self.NOW) is False

    def test_zero_elapsed_short_interval_not_due(self):
        source = make_source(poll_interval_minutes=1)
        # Polled 30 seconds ago, interval is 1 minute → not due yet.
        last = self.NOW - timedelta(seconds=30)
        assert is_due(source, last_polled_at=last, now=self.NOW) is False

    def test_different_poll_intervals_respected(self):
        now = self.NOW
        last = now - timedelta(minutes=3)
        short = make_source(poll_interval_minutes=2)   # 3 min elapsed > 2 min → due
        long_src = make_source(poll_interval_minutes=10)  # 3 min elapsed < 10 min → not due
        assert is_due(short, last_polled_at=last, now=now) is True
        assert is_due(long_src, last_polled_at=last, now=now) is False


# ---------------------------------------------------------------------------
# select_due_sources
# ---------------------------------------------------------------------------


class TestSelectDueSources:
    NOW = utc(2024, 6, 1, 12, 0, 0)

    def test_all_never_polled_all_returned(self):
        sources = [make_source("a"), make_source("b"), make_source("c")]
        store = FakePollStateStore()  # no state → all None
        due = select_due_sources(sources, store, self.NOW)
        assert [s.name for s in due] == ["a", "b", "c"]

    def test_none_due_returns_empty(self):
        sources = [make_source("a"), make_source("b")]
        # Polled 1 second ago, interval is 5 minutes.
        just_polled = PollState(last_polled_at=self.NOW - timedelta(seconds=1), etag=None, last_modified=None)
        store = FakePollStateStore({"a": just_polled, "b": just_polled})
        due = select_due_sources(sources, store, self.NOW)
        assert due == []

    def test_mixed_only_due_returned(self):
        sources = [make_source("due"), make_source("not-due")]
        overdue = PollState(last_polled_at=self.NOW - timedelta(hours=1), etag=None, last_modified=None)
        recent = PollState(last_polled_at=self.NOW - timedelta(seconds=10), etag=None, last_modified=None)
        store = FakePollStateStore({"due": overdue, "not-due": recent})
        due = select_due_sources(sources, store, self.NOW)
        assert [s.name for s in due] == ["due"]

    def test_disabled_source_never_returned(self):
        disabled = make_source("disabled", enabled=False)
        store = FakePollStateStore()  # no state → would be due if enabled
        due = select_due_sources([disabled], store, self.NOW)
        assert due == []

    def test_disabled_and_overdue_still_skipped(self):
        disabled = make_source("overdue-disabled", enabled=False)
        overdue = PollState(last_polled_at=self.NOW - timedelta(days=365), etag=None, last_modified=None)
        store = FakePollStateStore({"overdue-disabled": overdue})
        due = select_due_sources([disabled], store, self.NOW)
        assert due == []

    def test_enabled_and_disabled_mix(self):
        sources = [
            make_source("enabled-due"),
            make_source("disabled", enabled=False),
            make_source("enabled-not-due"),
        ]
        overdue = PollState(last_polled_at=self.NOW - timedelta(hours=1), etag=None, last_modified=None)
        recent = PollState(last_polled_at=self.NOW - timedelta(seconds=1), etag=None, last_modified=None)
        store = FakePollStateStore({"enabled-due": overdue, "enabled-not-due": recent})
        due = select_due_sources(sources, store, self.NOW)
        assert [s.name for s in due] == ["enabled-due"]

    def test_empty_sources_returns_empty(self):
        store = FakePollStateStore()
        assert select_due_sources([], store, self.NOW) == []

    def test_source_with_no_state_entry_is_treated_as_never_polled(self):
        source = make_source("fresh")
        store = FakePollStateStore({"other-source": PollState(last_polled_at=self.NOW, etag=None, last_modified=None)})
        due = select_due_sources([source], store, self.NOW)
        assert [s.name for s in due] == ["fresh"]


# ---------------------------------------------------------------------------
# tick
# ---------------------------------------------------------------------------


class TestTick:
    NOW = utc(2024, 6, 1, 12, 0, 0)

    def _overdue_store(self, *names: str) -> FakePollStateStore:
        """Return a store where the given sources have old poll timestamps (overdue)."""
        overdue = PollState(last_polled_at=self.NOW - timedelta(hours=1), etag=None, last_modified=None)
        return FakePollStateStore({name: overdue for name in names})

    def test_dispatches_due_source(self):
        source = make_source("s")
        store = FakePollStateStore()  # never polled → due
        engine = StubEngine()
        tick([source], store, engine, self.NOW)
        assert engine.processed == ["s"]

    def test_does_not_dispatch_non_due_source(self):
        source = make_source("s")
        recent = PollState(last_polled_at=self.NOW - timedelta(seconds=1), etag=None, last_modified=None)
        store = FakePollStateStore({"s": recent})
        engine = StubEngine()
        tick([source], store, engine, self.NOW)
        assert engine.processed == []

    def test_all_due_sources_dispatched(self):
        sources = [make_source("a"), make_source("b"), make_source("c")]
        store = FakePollStateStore()
        engine = StubEngine()
        tick(sources, store, engine, self.NOW)
        assert set(engine.processed) == {"a", "b", "c"}

    def test_source_failure_does_not_abort_tick(self):
        # If "bad" raises, "good" must still be processed.
        sources = [make_source("bad"), make_source("good")]
        store = FakePollStateStore()
        engine = StubEngine(raises_for={"bad"})
        tick(sources, store, engine, self.NOW)
        assert "good" in engine.processed

    def test_first_source_failure_does_not_skip_remaining(self):
        sources = [make_source("a"), make_source("b"), make_source("c")]
        store = FakePollStateStore()
        engine = StubEngine(raises_for={"a"})
        tick(sources, store, engine, self.NOW)
        # "a" was attempted (and failed), "b" and "c" were still processed.
        assert "a" in engine.processed
        assert "b" in engine.processed
        assert "c" in engine.processed

    def test_all_sources_fail_tick_still_completes(self):
        sources = [make_source("a"), make_source("b")]
        store = FakePollStateStore()
        engine = StubEngine(raises_for={"a", "b"})
        # Must not raise.
        tick(sources, store, engine, self.NOW)
        assert set(engine.processed) == {"a", "b"}

    def test_no_due_sources_engine_never_called(self):
        source = make_source("s")
        recent = PollState(last_polled_at=self.NOW - timedelta(seconds=1), etag=None, last_modified=None)
        store = FakePollStateStore({"s": recent})
        engine = StubEngine()
        tick([source], store, engine, self.NOW)
        assert engine.processed == []

    def test_empty_source_list_engine_never_called(self):
        store = FakePollStateStore()
        engine = StubEngine()
        tick([], store, engine, self.NOW)
        assert engine.processed == []

    def test_disabled_source_not_dispatched(self):
        source = make_source("s", enabled=False)
        store = FakePollStateStore()
        engine = StubEngine()
        tick([source], store, engine, self.NOW)
        assert engine.processed == []

    def test_order_of_dispatch_matches_registry_order(self):
        sources = [make_source("first"), make_source("second"), make_source("third")]
        store = FakePollStateStore()
        engine = StubEngine()
        tick(sources, store, engine, self.NOW)
        assert engine.processed == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# run_forever
# ---------------------------------------------------------------------------


class TestRunForever:
    def test_stop_event_set_before_start_exits_without_ticking(self):
        sources = [make_source("s")]
        store = FakePollStateStore()
        engine = StubEngine()
        stop_event = threading.Event()
        stop_event.set()  # already signalled before entering the loop
        run_forever(sources, store, engine, tick_interval=0.01, stop_event=stop_event)
        assert engine.processed == [], "No sources should be processed when stop_event is pre-set"

    def test_stop_event_set_after_one_tick_exits_cleanly(self):
        """Start the loop in a thread, let one tick fire, then signal shutdown."""
        sources = [make_source("s")]
        store = FakePollStateStore()
        engine = StubEngine()
        stop_event = threading.Event()

        thread = threading.Thread(
            target=run_forever,
            args=(sources, store, engine, 0.05, stop_event),
            daemon=True,
        )
        thread.start()

        # Give the first tick time to fire, then signal shutdown.
        # A brief sleep here is acceptable: we're testing process lifecycle,
        # which inherently involves real time.
        import time
        time.sleep(0.1)
        stop_event.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive(), "run_forever should exit within the timeout"
        assert "s" in engine.processed, "At least one tick should have run"

    def test_run_forever_exits_promptly_after_signal(self):
        """Shutdown latency should be much less than tick_interval."""
        import time
        sources = [make_source("s")]
        store = FakePollStateStore()
        engine = StubEngine()
        stop_event = threading.Event()

        # Use a long tick_interval to verify the interruptible sleep works.
        tick_interval = 10.0

        thread = threading.Thread(
            target=run_forever,
            args=(sources, store, engine, tick_interval, stop_event),
            daemon=True,
        )
        start = time.monotonic()
        thread.start()

        # Let the first tick run, then signal.
        time.sleep(0.05)
        stop_event.set()
        thread.join(timeout=2.0)
        elapsed = time.monotonic() - start

        assert not thread.is_alive()
        # Should exit in well under tick_interval (10 s); 2 s is a generous bound.
        assert elapsed < 2.0, (
            f"run_forever took {elapsed:.1f}s to stop — interruptible sleep not working"
        )
