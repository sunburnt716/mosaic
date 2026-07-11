"""Tests for ingestion/engine.py — ConcreteEngine.

ConcreteEngine.process_source(source) handles exactly ONE source and returns None (the
Engine Protocol contract); it never loops over multiple sources and never checks
`enabled`. Multi-source dispatch, disabled-source skipping, and per-source failure
isolation across a whole tick are the scheduler's job and are already covered in
test_run.py's TestTick (via a StubEngine). This file covers what ConcreteEngine itself
does for one source: the four dedup branches, the 304 short-circuit, fail-closed
transport rejection, per-record drop-and-count, and the poll-state update contract.

All storage uses in-memory SQLite; network calls are monkeypatched at the adapter's
_fetch_feed boundary. Since process_source returns None, outcomes are observed through
the stores' state after the call (and, where useful, through caplog).
"""

import logging
from datetime import datetime, timezone

import pytest

from ingestion.adapters.base import NotModifiedSignal, TransportError
from ingestion.adapters.rss import RssAdapter
from ingestion.engine import ConcreteEngine
from ingestion.pipeline.normalizer import normalize
from ingestion.storage.poll_state import PollStateStore
from ingestion.storage.raw_store import RawStore
from ingestion.storage.seen_store import SeenStore
from tests.conftest import load_fixture, make_source_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def stores(tmp_path):
    seen = SeenStore(":memory:")
    raw = RawStore(":memory:")
    poll = PollStateStore(tmp_path / "poll_state.json")
    yield seen, raw, poll
    seen.close()
    raw.close()


@pytest.fixture
def engine(stores):
    seen, raw, poll = stores
    return ConcreteEngine(raw, seen, poll)


@pytest.fixture
def rss_config():
    return make_source_config(name="test-reuters", adapter="rss", tier=1)


def _patch_rss(monkeypatch, items):
    """Monkeypatch RssAdapter._fetch_feed to return `items` without hitting the network."""
    monkeypatch.setattr(RssAdapter, "_fetch_feed", lambda self, url, headers: items)


def _patch_rss_raises(monkeypatch, exc: Exception):
    def _raise(self, url, headers):
        raise exc

    monkeypatch.setattr(RssAdapter, "_fetch_feed", _raise)


# ---------------------------------------------------------------------------
# Happy path — NEW document ingested end-to-end
# ---------------------------------------------------------------------------


class TestEngineNewDocument:
    def test_new_doc_saved_to_raw_store(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])
        engine.process_source(rss_config)

        doc = normalize(item, rss_config, datetime.now(timezone.utc))
        assert raw.get_document(doc.id) is not None

    def test_new_doc_registered_in_seen_store(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])
        engine.process_source(rss_config)

        ikey = f"{rss_config.name}::{item['source_article_id']}"
        stored_hash = seen.get_hash(ikey)
        assert stored_hash is not None
        assert seen.contains_hash(stored_hash)

    def test_poll_state_updated_after_run(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        _patch_rss(monkeypatch, [load_fixture("rss_reuters_sample.json")])
        engine.process_source(rss_config)
        state = poll.get(rss_config.name)
        assert state.last_polled_at is not None


# ---------------------------------------------------------------------------
# L1 — Exact duplicate: second call produces no new save
# ---------------------------------------------------------------------------


class TestEngineL1Duplicate:
    def test_second_call_does_not_duplicate_seen_entry(
        self, monkeypatch, stores, engine, rss_config
    ):
        seen, raw, poll = stores
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])
        engine.process_source(rss_config)  # first call: NEW
        engine.process_source(rss_config)  # second call: L1 — no error, no change

        ikey = f"{rss_config.name}::{item['source_article_id']}"
        # Still exactly the original hash; an L1 duplicate never rewrites the seen entry.
        doc = normalize(item, rss_config, datetime.now(timezone.utc))
        assert seen.get_hash(ikey) == doc.content_hash

    def test_l1_does_not_overwrite_raw_payload(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])
        engine.process_source(rss_config)

        # Tamper the fixture — but the raw_store should still hold the original.
        tampered = dict(item)
        tampered["raw_payload"] = {"tampered": True}
        _patch_rss(monkeypatch, [tampered])
        engine.process_source(rss_config)

        # get_raw uses doc_id which is derived from content; same content -> same id.
        doc = normalize(item, rss_config, datetime.now(timezone.utc))
        stored_raw = raw.get_raw(doc.id)
        assert stored_raw == item["raw_payload"]


# ---------------------------------------------------------------------------
# L2 — Updated article: second call with changed body
# ---------------------------------------------------------------------------


class TestEngineL2Update:
    def test_l2_update_overwrites_document(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        item_v1 = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item_v1])
        engine.process_source(rss_config)

        # Same source_article_id (same identity_key) but different body -> L2.
        item_v2 = dict(item_v1)
        item_v2["raw_body"] = "<p>Updated: Fed cuts rates by 50bps.</p>"
        item_v2["title"] = "UPDATED: Fed cuts rates"
        item_v2["raw_payload"] = dict(item_v1["raw_payload"])
        _patch_rss(monkeypatch, [item_v2])
        engine.process_source(rss_config)

        doc_v2 = normalize(item_v2, rss_config, datetime.now(timezone.utc))
        stored = raw.get_document(doc_v2.id)
        assert stored is not None
        assert stored.title == "UPDATED: Fed cuts rates"

    def test_l2_update_registers_new_hash(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        item_v1 = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item_v1])
        engine.process_source(rss_config)

        item_v2 = dict(item_v1)
        item_v2["raw_body"] = "<p>Updated: Fed cuts rates by 50bps instead.</p>"
        item_v2["raw_payload"] = dict(item_v1["raw_payload"])
        _patch_rss(monkeypatch, [item_v2])
        engine.process_source(rss_config)

        doc_v2 = normalize(item_v2, rss_config, datetime.now(timezone.utc))
        ikey = f"{rss_config.name}::{item_v1['source_article_id']}"
        assert seen.get_hash(ikey) == doc_v2.content_hash


# ---------------------------------------------------------------------------
# 304 Not Modified short-circuit
# ---------------------------------------------------------------------------


class TestEngine304ShortCircuit:
    def test_304_does_not_raise(self, monkeypatch, engine, rss_config):
        _patch_rss_raises(monkeypatch, NotModifiedSignal("304"))
        engine.process_source(rss_config)  # must not raise

    def test_304_updates_poll_state(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        _patch_rss_raises(monkeypatch, NotModifiedSignal("304"))
        engine.process_source(rss_config)
        state = poll.get(rss_config.name)
        assert state.last_polled_at is not None

    def test_304_preserves_previous_validators(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        # A prior successful poll recorded validators.
        resp_item = load_fixture("rss_reuters_sample.json")
        resp_item["_etag"] = '"v1"'
        _patch_rss(monkeypatch, [resp_item])
        engine.process_source(rss_config)
        assert poll.get(rss_config.name).etag == '"v1"'

        # A 304 on the next poll must not clear the stored etag.
        _patch_rss_raises(monkeypatch, NotModifiedSignal("304"))
        engine.process_source(rss_config)
        assert poll.get(rss_config.name).etag == '"v1"'

    def test_304_does_not_write_to_stores(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        _patch_rss_raises(monkeypatch, NotModifiedSignal("304"))
        engine.process_source(rss_config)
        assert raw.get_document("anything") is None


# ---------------------------------------------------------------------------
# Fetch failure — logged and swallowed, never raised (source isolation is tick()'s job)
# ---------------------------------------------------------------------------


class TestEngineFetchFailure:
    def test_fetch_error_does_not_raise(self, monkeypatch, engine, rss_config):
        _patch_rss_raises(monkeypatch, ConnectionError("DNS failure"))
        engine.process_source(rss_config)  # must not raise

    def test_fetch_error_still_touches_poll_state(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        _patch_rss_raises(monkeypatch, ConnectionError("DNS failure"))
        engine.process_source(rss_config)
        assert poll.get(rss_config.name).last_polled_at is not None

    def test_fetch_error_is_logged(self, monkeypatch, engine, rss_config, caplog):
        _patch_rss_raises(monkeypatch, ConnectionError("DNS failure"))
        with caplog.at_level(logging.ERROR):
            engine.process_source(rss_config)
        assert any("FetchError" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fail-closed transport rejection (whole batch refused)
# ---------------------------------------------------------------------------


class TestEngineTransportRejection:
    def test_transport_error_writes_nothing(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        _patch_rss_raises(monkeypatch, TransportError("HTML challenge page"))
        engine.process_source(rss_config)
        assert raw.get_document("anything") is None

    def test_transport_rejection_does_not_raise(self, monkeypatch, engine, rss_config):
        _patch_rss_raises(monkeypatch, TransportError("HTML challenge page"))
        engine.process_source(rss_config)  # must not raise

    def test_transport_rejection_still_touches_poll_state(
        self, monkeypatch, stores, engine, rss_config
    ):
        seen, raw, poll = stores
        _patch_rss_raises(monkeypatch, TransportError("bad"))
        engine.process_source(rss_config)
        assert poll.get(rss_config.name).last_polled_at is not None


# ---------------------------------------------------------------------------
# Per-record contract: drop the bad record, keep the batch
# ---------------------------------------------------------------------------


class TestEnginePerRecordDrop:
    def test_bad_record_dropped_good_record_kept(self, monkeypatch, stores, engine, rss_config):
        seen, raw, poll = stores
        good = load_fixture("rss_reuters_sample.json")
        bad = dict(good)
        bad["url"] = ""  # missing URL -> NormalizationError -> dropped, not fatal
        bad["raw_payload"] = {"id": "bad-1"}
        bad["source_article_id"] = "bad-1"

        _patch_rss(monkeypatch, [good, bad])
        engine.process_source(rss_config)  # must not raise despite the bad record

        good_doc = normalize(good, rss_config, datetime.now(timezone.utc))
        assert raw.get_document(good_doc.id) is not None
        # The bad record's identity_key must never reach the seen store.
        assert seen.get_hash(f"{rss_config.name}::bad-1") is None

    def test_malformed_url_record_is_dropped_good_kept(
        self, monkeypatch, stores, engine, rss_config
    ):
        seen, raw, poll = stores
        good = load_fixture("rss_reuters_sample.json")
        bad = dict(good)
        bad["url"] = "not-a-real-url"  # no scheme/netloc -> dropped
        bad["raw_payload"] = {"id": "bad-2"}
        bad["source_article_id"] = "bad-2"

        _patch_rss(monkeypatch, [bad, good])
        engine.process_source(rss_config)

        good_doc = normalize(good, rss_config, datetime.now(timezone.utc))
        assert raw.get_document(good_doc.id) is not None
        assert seen.get_hash(f"{rss_config.name}::bad-2") is None


# ---------------------------------------------------------------------------
# Hot-path extraction: processing_mode="hot" invokes the on_processed callback
# ---------------------------------------------------------------------------


class TestEngineHotPath:
    def test_hot_mode_source_invokes_on_processed(self, monkeypatch, stores):
        seen, raw, poll = stores
        calls = []
        engine = ConcreteEngine(raw, seen, poll, on_processed=calls.append)
        config = make_source_config(name="test-reuters", adapter="rss", processing_mode="hot")
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])

        engine.process_source(config)

        assert len(calls) == 1
        assert calls[0].id == normalize(item, config, datetime.now(timezone.utc)).id

    def test_cold_mode_source_never_invokes_on_processed(self, monkeypatch, stores):
        seen, raw, poll = stores
        calls = []
        engine = ConcreteEngine(raw, seen, poll, on_processed=calls.append)
        config = make_source_config(name="test-reuters", adapter="rss", processing_mode="cold")
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])

        engine.process_source(config)

        assert calls == []

    def test_no_on_processed_configured_hot_mode_is_noop(self, monkeypatch, stores):
        # engine fixture has no on_processed wired; hot mode must not raise.
        seen, raw, poll = stores
        engine = ConcreteEngine(raw, seen, poll)
        config = make_source_config(name="test-reuters", adapter="rss", processing_mode="hot")
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])

        engine.process_source(config)  # must not raise

        doc = normalize(item, config, datetime.now(timezone.utc))
        assert raw.get_document(doc.id) is not None

    def test_on_processed_failure_is_caught_and_counted(self, monkeypatch, stores, caplog):
        seen, raw, poll = stores

        def _boom(doc):
            raise RuntimeError("simulated extraction failure")

        engine = ConcreteEngine(raw, seen, poll, on_processed=_boom)
        config = make_source_config(name="test-reuters", adapter="rss", processing_mode="hot")
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])

        with caplog.at_level(logging.ERROR):
            engine.process_source(config)  # must not raise despite on_processed failing

        # The document is still stored — a hot-path failure doesn't undo ingestion.
        doc = normalize(item, config, datetime.now(timezone.utc))
        assert raw.get_document(doc.id) is not None
        assert "Hot-path extraction failed" in caplog.text

    def test_on_processed_not_called_for_l1_duplicate(self, monkeypatch, stores):
        seen, raw, poll = stores
        calls = []
        engine = ConcreteEngine(raw, seen, poll, on_processed=calls.append)
        config = make_source_config(name="test-reuters", adapter="rss", processing_mode="hot")
        item = load_fixture("rss_reuters_sample.json")
        _patch_rss(monkeypatch, [item])

        engine.process_source(config)  # first pass: NEW, extracted
        calls.clear()
        engine.process_source(config)  # second pass: exact L1 duplicate

        assert calls == []
