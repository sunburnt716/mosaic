"""Contract tests for ingestion/pipeline/body_enrichment.py.

Offline: an injected fake `fetch_url` serves captured EDGAR fixtures (index page + primary
document), so the two-hop resolve/fetch/clean path is exercised without the network. Pins
the load-bearing guarantees: the real filing text replaces the feed snippet, and any failure
falls back to the snippet rather than dropping the record.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.pipeline.body_enrichment import (
    DEFAULT_MAX_BODY_CHARS,
    _resolve_primary_document,
    enrich_body,
)
from tests.conftest import make_source_config

_FIXTURES = Path(__file__).parent / "fixtures"
_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/1083446/000110465926082892/"
    "0001104659-26-082892-index.htm"
)
_PRIMARY_URL = (
    "https://www.sec.gov/Archives/edgar/data/1083446/000110465926082892/tm2620111d1_8k.htm"
)


def _index_html() -> str:
    return (_FIXTURES / "edgar-index.html").read_text()


def _primary_html() -> str:
    return (_FIXTURES / "edgar-8k.html").read_text()


class _FakeFetcher:
    """Serves the captured EDGAR fixtures by URL; records what it was asked for."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping
        self.calls: list[str] = []

    def __call__(self, url: str, headers: dict) -> str:
        self.calls.append(url)
        if url not in self._mapping:
            raise AssertionError(f"unexpected fetch: {url}")
        return self._mapping[url]


def _edgar_config():
    return make_source_config(
        name="sec-edgar",
        tier=0,
        doc_type="filing",
        body_fetch="edgar_filing",
        headers={"User-Agent": "MosaicRAG test@example.com"},
    )


def _edgar_raw():
    return {
        "url": _INDEX_URL,
        "title": "8-K — Astrana Health, Inc.",
        "raw_body": "8-K — Astrana Health, Inc.",  # the thin feed snippet
        "published": "2026-07-11T00:00:00Z",
        "source_article_id": _INDEX_URL,
        "raw_payload": {"link": _INDEX_URL},
    }


class TestEdgarFilingStrategy:
    def test_replaces_snippet_with_real_filing_text(self):
        fetch = _FakeFetcher({_INDEX_URL: _index_html(), _PRIMARY_URL: _primary_html()})
        out = enrich_body(_edgar_raw(), _edgar_config(), fetch_url=fetch)
        body = out["raw_body"]
        # The material Item 1.01 text is now present — what a claim can be grounded in.
        assert "Item 1.01" in body
        assert "material definitive agreement" in body.lower()
        assert "$745 million" in body
        # Two-hop: index first, then the primary document (not the exhibit).
        assert fetch.calls == [_INDEX_URL, _PRIMARY_URL]

    def test_strips_script_and_style_from_filing(self):
        fetch = _FakeFetcher({_INDEX_URL: _index_html(), _PRIMARY_URL: _primary_html()})
        out = enrich_body(_edgar_raw(), _edgar_config(), fetch_url=fetch)
        assert "window.__filing" not in out["raw_body"]  # <script> body dropped
        assert "page-break-after" not in out["raw_body"]  # <style> body dropped

    def test_original_dict_not_mutated(self):
        raw = _edgar_raw()
        fetch = _FakeFetcher({_INDEX_URL: _index_html(), _PRIMARY_URL: _primary_html()})
        enrich_body(raw, _edgar_config(), fetch_url=fetch)
        assert raw["raw_body"] == "8-K — Astrana Health, Inc."  # unchanged in place

    def test_body_capped_at_max_chars(self):
        huge = "<p>" + ("word " * 40_000) + "</p>"  # ~200k chars of text
        fetch = _FakeFetcher({_INDEX_URL: _index_html(), _PRIMARY_URL: huge})
        out = enrich_body(_edgar_raw(), _edgar_config(), fetch_url=fetch)
        assert len(out["raw_body"]) <= DEFAULT_MAX_BODY_CHARS


class TestBestEffortFallback:
    def test_fetch_failure_keeps_feed_snippet(self):
        def boom(url, headers):
            raise ConnectionError("network down")

        out = enrich_body(_edgar_raw(), _edgar_config(), fetch_url=boom)
        assert out["raw_body"] == "8-K — Astrana Health, Inc."  # fell back, did not raise

    def test_index_without_primary_document_keeps_snippet(self):
        fetch = _FakeFetcher({_INDEX_URL: "<html><body>no documents here</body></html>"})
        out = enrich_body(_edgar_raw(), _edgar_config(), fetch_url=fetch)
        assert out["raw_body"] == "8-K — Astrana Health, Inc."

    def test_empty_extracted_body_keeps_snippet(self):
        fetch = _FakeFetcher({_INDEX_URL: _index_html(), _PRIMARY_URL: "<html></html>"})
        out = enrich_body(_edgar_raw(), _edgar_config(), fetch_url=fetch)
        assert out["raw_body"] == "8-K — Astrana Health, Inc."


class TestOptOut:
    def test_no_body_fetch_returns_unchanged_without_fetching(self):
        def must_not_be_called(url, headers):
            raise AssertionError("fetch_url must not run when body_fetch is None")

        config = make_source_config(name="ft", body_fetch=None)
        raw = {"url": "https://ft.com/x", "raw_body": "standfirst"}
        out = enrich_body(raw, config, fetch_url=must_not_be_called)
        assert out is raw

    def test_unknown_strategy_falls_back_unchanged(self):
        config = make_source_config(name="x", body_fetch="does_not_exist")
        raw = {"url": "https://x/y", "raw_body": "snippet"}
        out = enrich_body(raw, config, fetch_url=lambda u, h: "should not matter")
        assert out["raw_body"] == "snippet"


class TestResolvePrimaryDocument:
    def test_picks_first_document_format_file_not_the_index(self):
        assert _resolve_primary_document(_index_html(), _INDEX_URL) == _PRIMARY_URL

    def test_returns_none_when_no_document_links(self):
        html = '<html><a href="/cgi-bin/browse-edgar?action=getcompany">filer</a></html>'
        assert _resolve_primary_document(html, _INDEX_URL) is None
