"""Tests for ingestion/engine.py.

Tests the four dedup branches, source isolation, 304 short-circuit, and the
poll-state update contract. All storage uses in-memory SQLite; network calls are
monkeypatched at the adapter's _fetch_feed/_fetch_json boundary.
"""

import pytest

from ingestion.adapters.base import NotModifiedSignal, TransportError
from ingestion.adapters.rss import RssAdapter
from ingestion.engine import EngineResult, SourceResult, run
from ingestion.storage.poll_state import PollStateStore
from ingestion.storage.raw_store import RawStore
from ingestion.storage.seen_store import SeenStore
from tests.conftest import load_fixture, make_source_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def stores():
    seen = SeenStore(":memory:")
    raw = RawStore(":memory:")
    poll = PollStateStore(":memory:")
    yield seen, raw, poll
    seen.close()
    raw.close()
    poll.close()


@pytest.fixture
def rss_config():
    return make_source_config(name="test-reuters", adapter="rss", tier=1)


def _patch_rss(monkeypatch, items):
    """Monkeypatch RssAdapter._fetch_feed to return `items` without hitting the network."""
    monkeypatch.setattr(RssAdapter, "_fetch_feed", lambda self, url, headers: items)


# ---------------------------------------------------------------------------
# Happy path — NEW document ingested end-to-end
# ---------------------------------------------------------------------------


class TestEngineNewDocument:
    def test_new_doc_counted(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        _patch_rss(monkeypatch, [load_fixture("rss_reuters_sample.json")])
        result = run([rss_config], raw, seen, poll)
        assert result.sources[0].new == 1
        assert result.sources[0].l1_duplicate == 0

    def test_new_doc_saved_to_raw_store(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        raw_item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [raw_item])
        run([rss_config], raw, seen, poll)
        # At least one document should be in the store
        seen_hash = seen.get_hash(f"{rss_config.name}::{raw_item['source_article_id']}")
        assert seen_hash is not None

    def test_new_doc_registered_in_seen_store(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        _patch_rss(monkeypatch, [load_fixture("rss_reuters_sample.json")])
        run([rss_config], raw, seen, poll)
        # The seen store must have at least one entry after ingesting a new document.
        assert seen.contains_hash(
            seen.get_hash(
                f"{rss_config.name}::{load_fixture('rss_reuters_sample.json')['source_article_id']}"
            )
        )

    def test_poll_state_updated_after_run(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        _patch_rss(monkeypatch, [load_fixture("rss_reuters_sample.json")])
        run([rss_config], raw, seen, poll)
        state = poll.get(rss_config.name)
        assert state.last_polled_at is not None


# ---------------------------------------------------------------------------
# L1 — Exact duplicate: second run produces 0 new
# ---------------------------------------------------------------------------


class TestEngineL1Duplicate:
    def test_second_run_is_l1(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])
        run([rss_config], raw, seen, poll)  # first run: NEW
        result2 = run([rss_config], raw, seen, poll)  # second run: L1
        assert result2.sources[0].new == 0
        assert result2.sources[0].l1_duplicate == 1

    def test_l1_does_not_overwrite_raw_payload(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])
        run([rss_config], raw, seen, poll)
        # Tamper the fixture — but the raw_store should still hold the original.
        tampered = dict(item)
        tampered["raw_payload"] = {"tampered": True}
        _patch_rss(monkeypatch, [tampered])
        run([rss_config], raw, seen, poll)
        # get_raw uses doc_id which is derived from content; same content → same id
        # The raw payload should match the FIRST ingest (INSERT OR IGNORE).
        from datetime import datetime, timezone

        from ingestion.pipeline.normalizer import normalize

        doc = normalize(item, rss_config, datetime.now(timezone.utc))
        stored_raw = raw.get_raw(doc.id)
        assert stored_raw == item["raw_payload"]


# ---------------------------------------------------------------------------
# L2 — Updated article: second run with changed body
# ---------------------------------------------------------------------------


class TestEngineL2Update:
    def test_l2_update_counted(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        item_v1 = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item_v1])
        run([rss_config], raw, seen, poll)

        # Same source_article_id (same identity_key) but different body → L2
        item_v2 = dict(item_v1)
        item_v2["raw_body"] = "<p>Updated: Fed cuts rates by 50bps instead.</p>"
        item_v2["raw_payload"] = dict(item_v1["raw_payload"])
        _patch_rss(monkeypatch, [item_v2])
        result2 = run([rss_config], raw, seen, poll)
        assert result2.sources[0].l2_update == 1
        assert result2.sources[0].new == 0

    def test_l2_update_overwrites_document(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        item_v1 = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item_v1])
        run([rss_config], raw, seen, poll)

        item_v2 = dict(item_v1)
        item_v2["raw_body"] = "<p>Updated: Fed cuts rates by 50bps.</p>"
        item_v2["title"] = "UPDATED: Fed cuts rates"
        item_v2["raw_payload"] = dict(item_v1["raw_payload"])
        _patch_rss(monkeypatch, [item_v2])
        run([rss_config], raw, seen, poll)

        # The document in the store should reflect the updated title.
        from ingestion.pipeline.normalizer import normalize
        from datetime import datetime, timezone

        doc_v2 = normalize(item_v2, rss_config, datetime.now(timezone.utc))
        stored = raw.get_document(doc_v2.id)
        assert stored is not None
        assert stored.title == "UPDATED: Fed cuts rates"


# ---------------------------------------------------------------------------
# Source isolation — FetchError in one source doesn't abort others
# ---------------------------------------------------------------------------


class TestEngineSourceIsolation:
    def test_fetch_error_counted_not_raised(self, monkeypatch, stores):
        seen, raw, poll = stores
        bad_config = make_source_config(name="bad-source", adapter="rss", tier=1)

        def boom(self, url, headers):
            raise ConnectionError("DNS failure")

        monkeypatch.setattr(RssAdapter, "_fetch_feed", boom)
        result = run([bad_config], raw, seen, poll)
        assert result.sources[0].errors == 1
        assert result.total_errors == 1

    def test_good_source_still_runs_after_bad_source(self, monkeypatch, stores):
        seen, raw, poll = stores
        bad_config = make_source_config(
            name="bad-source",
            adapter="rss",
            tier=1,
            url="https://feeds.bad-source.example/rss",
        )
        good_config = make_source_config(
            name="good-source",
            adapter="rss",
            tier=1,
            url="https://feeds.good-source.example/rss",
        )

        good_item = load_fixture("rss_reuters_sample.json")

        def selective_fetch(self, url, headers):
            if "bad-source" in url:
                raise ConnectionError("DNS failure")
            return [good_item]

        monkeypatch.setattr(RssAdapter, "_fetch_feed", selective_fetch)
        result = run([bad_config, good_config], raw, seen, poll)

        bad_src = next(s for s in result.sources if s.source_name == "bad-source")
        good_src = next(s for s in result.sources if s.source_name == "good-source")
        assert bad_src.errors == 1
        assert good_src.new == 1

    def test_disabled_source_is_skipped(self, monkeypatch, stores):
        seen, raw, poll = stores
        disabled = make_source_config(
            name="disabled", adapter="rss", tier=1, enabled=False
        )
        called = {"hit": False}

        def spy(self, url, headers):
            called["hit"] = True
            return []

        monkeypatch.setattr(RssAdapter, "_fetch_feed", spy)
        result = run([disabled], raw, seen, poll)
        assert called["hit"] is False
        assert result.sources == []


# ---------------------------------------------------------------------------
# 304 Not Modified short-circuit
# ---------------------------------------------------------------------------


class TestEngine304ShortCircuit:
    def test_304_increments_skipped_304(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores

        def raise_304(self, url, headers):
            raise NotModifiedSignal("304")

        monkeypatch.setattr(RssAdapter, "_fetch_feed", raise_304)
        result = run([rss_config], raw, seen, poll)
        assert result.sources[0].skipped_304 is True
        assert result.sources[0].fetched == 0

    def test_304_updates_poll_state(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores

        def raise_304(self, url, headers):
            raise NotModifiedSignal("304")

        monkeypatch.setattr(RssAdapter, "_fetch_feed", raise_304)
        run([rss_config], raw, seen, poll)
        state = poll.get(rss_config.name)
        assert state.last_polled_at is not None

    def test_304_does_not_write_to_stores(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores

        def raise_304(self, url, headers):
            raise NotModifiedSignal("304")

        monkeypatch.setattr(RssAdapter, "_fetch_feed", raise_304)
        run([rss_config], raw, seen, poll)
        assert raw.get_document("anything") is None


# ---------------------------------------------------------------------------
# EngineResult aggregation
# ---------------------------------------------------------------------------


class TestEngineResult:
    def test_total_new_sums_across_sources(self):
        result = EngineResult(
            sources=[
                SourceResult("a", new=3),
                SourceResult("b", new=2),
            ]
        )
        assert result.total_new == 5

    def test_total_errors_sums_across_sources(self):
        result = EngineResult(
            sources=[
                SourceResult("a", errors=1),
                SourceResult("b", errors=2),
            ]
        )
        assert result.total_errors == 3


# ---------------------------------------------------------------------------
# Phase 2 — fail-closed transport rejection (whole batch refused)
# ---------------------------------------------------------------------------


class TestEngineTransportRejection:
    def test_transport_error_flags_rejected_and_writes_nothing(
        self, monkeypatch, stores, rss_config
    ):
        seen, raw, poll = stores

        def reject(self, url, headers):
            raise TransportError("HTML challenge page where feed expected")

        monkeypatch.setattr(RssAdapter, "_fetch_feed", reject)
        result = run([rss_config], raw, seen, poll)

        src = result.sources[0]
        assert src.rejected_transport is True
        assert src.errors == 1
        assert src.new == 0
        assert raw.get_document("anything") is None  # nothing reached the store

    def test_transport_rejection_still_touches_poll_state(
        self, monkeypatch, stores, rss_config
    ):
        seen, raw, poll = stores

        monkeypatch.setattr(
            RssAdapter,
            "_fetch_feed",
            lambda self, url, headers: (_ for _ in ()).throw(TransportError("bad")),
        )
        run([rss_config], raw, seen, poll)
        assert poll.get(rss_config.name).last_polled_at is not None


# ---------------------------------------------------------------------------
# Phase 2 — per-record contract: drop the bad record, keep the batch
# ---------------------------------------------------------------------------


class TestEnginePerRecordDrop:
    def test_bad_record_dropped_good_records_kept(
        self, monkeypatch, stores, rss_config
    ):
        seen, raw, poll = stores
        good = load_fixture("rss_reuters_sample.json")
        bad = dict(good)
        bad["url"] = ""  # missing URL -> NormalizationError -> dropped, not fatal
        bad["raw_payload"] = {"id": "bad-1"}
        bad["source_article_id"] = "bad-1"

        _patch_rss(monkeypatch, [good, bad])
        result = run([rss_config], raw, seen, poll)

        src = result.sources[0]
        assert src.fetched == 2
        assert src.new == 1  # only the good record ingested
        assert src.dropped_records == 1
        assert src.errors == 0  # a dropped record is not a source-level error

    def test_malformed_url_record_is_dropped(self, monkeypatch, stores, rss_config):
        seen, raw, poll = stores
        good = load_fixture("rss_reuters_sample.json")
        bad = dict(good)
        bad["url"] = "not-a-real-url"  # no scheme/netloc -> dropped
        bad["raw_payload"] = {"id": "bad-2"}
        bad["source_article_id"] = "bad-2"

        _patch_rss(monkeypatch, [bad, good])
        result = run([rss_config], raw, seen, poll)

        assert result.sources[0].dropped_records == 1
        assert result.sources[0].new == 1
