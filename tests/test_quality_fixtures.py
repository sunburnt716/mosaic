"""Fixture-backed regression tests for the quality gate (Phase 4), both directions.

These pin the gate's behavior against real/representative payloads run through the actual
adapter → normalize → gate chain, fully offline (requests.get is monkeypatched to return
captured bytes). They are the gate's regression suite — there is no separate harness.

  - degenerate_feed.xml (synthetic) → the gate WARNS (collapse + empty body + fallback title)
  - sec-edgar.xml (captured live) → the gate is SILENT (a healthy EDGAR batch)
"""

from datetime import datetime, timezone

import requests

from ingestion.adapters.rss import RssAdapter
from ingestion.pipeline.normalizer import normalize
from ingestion.pipeline.quality import check
from tests.conftest import FakeResponse, load_fixture_bytes, make_source_config

_FETCHED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _normalized_batch(fixture_name, config, monkeypatch):
    """Run a fixture's raw bytes through the real adapter + normalizer, return the batch.

    Drives RssAdapter._fetch_feed via a monkeypatched requests.get so the actual
    conditional-GET, transport, parse, and field-mapping code executes offline.
    """
    resp = FakeResponse(
        status_code=200,
        headers={"Content-Type": "application/rss+xml"},
        content=load_fixture_bytes(fixture_name),
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: resp)
    raws = list(RssAdapter().fetch(config))
    return [normalize(r, config, _FETCHED_AT) for r in raws]


class TestGateWarnsOnDegenerateBatch:
    def test_degenerate_feed_raises_warnings(self, monkeypatch):
        config = make_source_config(
            name="degenerate", expects={"title": True, "url": True, "body": True}
        )
        batch = _normalized_batch("degenerate_feed.xml", config, monkeypatch)
        report = check(batch, config)
        flags = " ".join(report.warnings)

        # The whole batch collapsed to one link with empty bodies + placeholder titles.
        assert "URL_COLLAPSE" in flags
        assert "IDENTITY_COLLAPSE" in flags
        assert "BODY_EMPTY" in flags
        assert "TITLE_FALLBACK" in flags

    def test_gate_does_not_drop_the_degenerate_batch(self, monkeypatch):
        """Soft contract: warnings fire, but every record is still present + counted."""
        config = make_source_config(name="degenerate")
        batch = _normalized_batch("degenerate_feed.xml", config, monkeypatch)
        report = check(batch, config)
        assert report.stats["records"] == len(batch) == 6


class TestGateSilentOnHealthyEdgar:
    def test_healthy_edgar_batch_no_warnings(self, monkeypatch):
        # EDGAR config mirrors ingestion/config/sources.yaml: filing doc_type, transform,
        # body not expected.
        config = make_source_config(
            name="sec-edgar",
            tier=0,
            doc_type="filing",
            transform="edgar_filing_url",
            expects={"title": True, "url": True, "body": False},
        )
        batch = _normalized_batch("sec-edgar.xml", config, monkeypatch)
        report = check(batch, config)

        assert report.warnings == []
        assert report.stats["records"] == len(batch)
        # Healthy batch: distinct citations, no collapse.
        assert report.stats["unique_urls"] == len(batch)
        assert report.stats["unique_identity_keys"] == len(batch)
