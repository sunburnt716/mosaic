"""Integration tests — hit live sources and validate the raw schema they return.

These tests make real network requests. They are skipped unless you set the
RUN_INTEGRATION env var:

    RUN_INTEGRATION=1 pytest -m integration -v          # all sources
    RUN_INTEGRATION=1 pytest -m integration -v -k Edgar  # one source

Each enabled source in config/sources.json gets its own test class. The assertions
check schema shape only: that required fields are present, have the right types,
and produce a valid normalized Document. Content values are not asserted since
they change with every poll.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from ingestion.adapters.registry import get_adapter
from ingestion.pipeline.normalizer import normalize
from ingestion.sources import DEFAULT_REGISTRY_PATH, load_sources

_SKIP_REASON = (
    "Requires live network access — set RUN_INTEGRATION=1 to run"
)
_skip_unless_live = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1", reason=_SKIP_REASON
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_RAW_FIELDS = {"url", "title", "raw_body", "published", "raw_payload"}
_SOURCES = {c.name: c for c in load_sources(DEFAULT_REGISTRY_PATH) if c.enabled}


def _assert_raw_item_schema(item: dict, source_name: str) -> None:
    """Assert that a raw adapter item has all required fields with sane types."""
    missing = REQUIRED_RAW_FIELDS - item.keys()
    assert not missing, f"[{source_name}] raw item missing fields: {missing}"
    assert isinstance(item["url"], str) and item["url"].startswith(
        "http"
    ), f"[{source_name}] url is not a valid http(s) URL: {item['url']!r}"
    assert isinstance(item["title"], str), f"[{source_name}] title is not a str"
    assert isinstance(item["raw_body"], str), f"[{source_name}] raw_body is not a str"
    assert isinstance(
        item["raw_payload"], dict
    ), f"[{source_name}] raw_payload is not a dict"


def _assert_document_schema(doc, source_name: str) -> None:
    """Assert that a normalized Document has the expected field types."""
    assert isinstance(doc.id, str) and len(doc.id) > 0
    assert isinstance(doc.url, str) and doc.url.startswith("http")
    assert isinstance(doc.title, str)
    assert isinstance(doc.body, str)
    assert isinstance(doc.tier, int)
    assert isinstance(doc.doc_type, str) and doc.doc_type in ("article", "filing")
    assert isinstance(doc.content_hash, str) and len(doc.content_hash) == 64
    assert isinstance(doc.identity_key, str)
    assert isinstance(doc.published_date, datetime) and doc.published_date.tzinfo is not None, (
        f"[{source_name}] published_date must be timezone-aware"
    )
    assert isinstance(doc.fetched_at, datetime) and doc.fetched_at.tzinfo is not None
    assert isinstance(doc.raw_payload, dict)
    assert isinstance(doc.source_name, str)
    assert doc.tier == _SOURCES[source_name].tier, (
        f"[{source_name}] tier mismatch: config says {_SOURCES[source_name].tier}, "
        f"document has {doc.tier}"
    )


# ---------------------------------------------------------------------------
# Per-source integration test classes
# ---------------------------------------------------------------------------


@pytest.mark.integration
@_skip_unless_live
class TestSecEdgarLive:
    """Live schema check for the SEC EDGAR 8-K getcurrent feed (Tier 0)."""

    SOURCE_NAME = "sec-edgar"

    def _config(self):
        return _SOURCES[self.SOURCE_NAME]

    def test_source_is_registered(self):
        assert self.SOURCE_NAME in _SOURCES, (
            f"{self.SOURCE_NAME!r} not found in sources.json or is disabled"
        )

    def test_adapter_fetches_items(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        items = list(adapter.fetch(config))
        assert len(items) > 0, f"[{self.SOURCE_NAME}] fetch returned zero items"

    def test_raw_item_schema(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        items = list(adapter.fetch(config))
        for item in items[:5]:
            _assert_raw_item_schema(item, self.SOURCE_NAME)

    def test_items_normalize_to_valid_documents(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        fetched_at = datetime.now(timezone.utc)
        items = list(adapter.fetch(config))
        assert items, f"[{self.SOURCE_NAME}] no items to normalize"
        for item in items[:3]:
            doc = normalize(item, config, fetched_at)
            _assert_document_schema(doc, self.SOURCE_NAME)

    def test_edgar_urls_are_archive_links(self):
        """EDGAR items must carry canonical archive URLs, not the feed landing page."""
        config = self._config()
        adapter = get_adapter(config.adapter)()
        items = list(adapter.fetch(config))
        fetched_at = datetime.now(timezone.utc)
        for item in items[:5]:
            doc = normalize(item, config, fetched_at)
            assert doc.url.startswith("https://www.sec.gov/Archives/edgar/"), (
                f"[{self.SOURCE_NAME}] URL is not a canonical EDGAR archive link: {doc.url!r}"
            )
            assert "[" not in doc.url and "]" not in doc.url, (
                f"[{self.SOURCE_NAME}] URL contains list repr artefact: {doc.url!r}"
            )

    def test_doc_type_is_filing(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        fetched_at = datetime.now(timezone.utc)
        items = list(adapter.fetch(config))
        for item in items[:3]:
            doc = normalize(item, config, fetched_at)
            assert doc.doc_type == "filing", (
                f"[{self.SOURCE_NAME}] expected doc_type=filing, got {doc.doc_type!r}"
            )


@pytest.mark.integration
@_skip_unless_live
class TestCnbcFinanceLive:
    """Live schema check for the CNBC Finance RSS feed (Tier 1)."""

    SOURCE_NAME = "cnbc-finance"

    def _config(self):
        return _SOURCES[self.SOURCE_NAME]

    def test_source_is_registered(self):
        assert self.SOURCE_NAME in _SOURCES

    def test_adapter_fetches_items(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        items = list(adapter.fetch(config))
        assert len(items) > 0, f"[{self.SOURCE_NAME}] fetch returned zero items"

    def test_raw_item_schema(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        items = list(adapter.fetch(config))
        for item in items[:5]:
            _assert_raw_item_schema(item, self.SOURCE_NAME)

    def test_items_normalize_to_valid_documents(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        fetched_at = datetime.now(timezone.utc)
        items = list(adapter.fetch(config))
        assert items, f"[{self.SOURCE_NAME}] no items to normalize"
        for item in items[:3]:
            doc = normalize(item, config, fetched_at)
            _assert_document_schema(doc, self.SOURCE_NAME)

    def test_doc_type_is_article(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        fetched_at = datetime.now(timezone.utc)
        items = list(adapter.fetch(config))
        for item in items[:3]:
            doc = normalize(item, config, fetched_at)
            assert doc.doc_type == "article", (
                f"[{self.SOURCE_NAME}] expected doc_type=article, got {doc.doc_type!r}"
            )

    def test_tier_is_stamped_correctly(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        fetched_at = datetime.now(timezone.utc)
        items = list(adapter.fetch(config))
        for item in items[:3]:
            doc = normalize(item, config, fetched_at)
            assert doc.tier == 1


@pytest.mark.integration
@_skip_unless_live
class TestFtRssLive:
    """Live schema check for the Financial Times RSS feed (Tier 2)."""

    SOURCE_NAME = "ft-rss"

    def _config(self):
        return _SOURCES[self.SOURCE_NAME]

    def test_source_is_registered(self):
        assert self.SOURCE_NAME in _SOURCES

    def test_adapter_fetches_items(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        items = list(adapter.fetch(config))
        assert len(items) > 0, f"[{self.SOURCE_NAME}] fetch returned zero items"

    def test_raw_item_schema(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        items = list(adapter.fetch(config))
        for item in items[:5]:
            _assert_raw_item_schema(item, self.SOURCE_NAME)

    def test_items_normalize_to_valid_documents(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        fetched_at = datetime.now(timezone.utc)
        items = list(adapter.fetch(config))
        assert items, f"[{self.SOURCE_NAME}] no items to normalize"
        for item in items[:3]:
            doc = normalize(item, config, fetched_at)
            _assert_document_schema(doc, self.SOURCE_NAME)

    def test_tier_is_stamped_correctly(self):
        config = self._config()
        adapter = get_adapter(config.adapter)()
        fetched_at = datetime.now(timezone.utc)
        items = list(adapter.fetch(config))
        for item in items[:3]:
            doc = normalize(item, config, fetched_at)
            assert doc.tier == 2
