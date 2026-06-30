"""Tests that every adapter honours the Adapter base contract from adapters/base.py.

Contract under test:
  Adapter.fetch(config: SourceConfig) -> Iterable[dict]
    - Yields one dict per logical document (one article, one filing)
    - Each dict contains at minimum: url, published (raw timestamp), raw_body, raw_payload
    - Raises FetchError (not bare Exception) on network or parse failure
    - Is stateless: all config comes from SourceConfig

  FetchError:
    - Raised (not bare Exception) on HTTP errors, malformed feed, or empty feed

Adding a new source adapter: implement Adapter, add it to adapters/registry.py,
add a fixture in tests/fixtures/, and add a test class here following the pattern below.
Do NOT add per-source code paths elsewhere — config only.
"""

from pathlib import Path

import pytest
import requests

from ingestion.adapters.base import (
    Adapter,
    FetchError,
    NotModifiedSignal,
    TransportError,
)
from ingestion.adapters.rest_json import RestJsonAdapter
from ingestion.adapters.rss import RssAdapter
from tests.conftest import FakeResponse, load_fixture, make_source_config

_CHALLENGE_HTML = (
    Path(__file__).parent / "fixtures" / "challenge_page.html"
).read_bytes()


# A tiny valid Atom document with one entry, so feedparser yields exactly one item.
_ATOM_ONE_ENTRY = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test entry</title>
    <link href="https://example.test/article/1"/>
    <id>urn:example:1</id>
    <updated>2026-01-01T00:00:00Z</updated>
    <summary>A summary.</summary>
  </entry>
</feed>"""


# ---------------------------------------------------------------------------
# Adapter base class contract
# ---------------------------------------------------------------------------


class TestAdapterBaseClass:
    def test_adapter_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            Adapter()

    def test_fetch_is_abstract(self):
        # Any concrete subclass that does not implement fetch() should also raise TypeError
        class IncompleteAdapter(Adapter):
            pass

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_fetch_error_is_exception(self):
        assert issubclass(FetchError, Exception)


# ---------------------------------------------------------------------------
# RSS adapter contract
# ---------------------------------------------------------------------------


class TestRssAdapterContract:
    def test_rss_adapter_is_adapter_subclass(self):
        assert issubclass(RssAdapter, Adapter)

    def test_fetch_yields_dicts(self, monkeypatch, reuters_source_config):
        raw = load_fixture("rss_reuters_sample.json")
        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [raw])

        results = list(adapter.fetch(reuters_source_config))
        assert len(results) >= 1
        assert all(isinstance(item, dict) for item in results)

    def test_each_item_has_url(self, monkeypatch, reuters_source_config):
        raw = load_fixture("rss_reuters_sample.json")
        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [raw])

        for item in adapter.fetch(reuters_source_config):
            assert "url" in item
            assert item["url"]

    def test_each_item_has_published_timestamp(
        self, monkeypatch, reuters_source_config
    ):
        raw = load_fixture("rss_reuters_sample.json")
        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [raw])

        for item in adapter.fetch(reuters_source_config):
            assert "published" in item

    def test_each_item_has_raw_body(self, monkeypatch, reuters_source_config):
        raw = load_fixture("rss_reuters_sample.json")
        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [raw])

        for item in adapter.fetch(reuters_source_config):
            assert "raw_body" in item

    def test_each_item_has_raw_payload(self, monkeypatch, reuters_source_config):
        raw = load_fixture("rss_reuters_sample.json")
        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [raw])

        for item in adapter.fetch(reuters_source_config):
            assert "raw_payload" in item
            assert isinstance(item["raw_payload"], dict)

    def test_raw_payload_is_untouched_source_response(
        self, monkeypatch, reuters_source_config
    ):
        raw = load_fixture("rss_reuters_sample.json")
        expected_payload = dict(raw["raw_payload"])
        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [raw])

        for item in adapter.fetch(reuters_source_config):
            assert item["raw_payload"] == expected_payload

    def test_network_failure_raises_fetch_error(
        self, monkeypatch, reuters_source_config
    ):
        adapter = RssAdapter()
        monkeypatch.setattr(
            adapter,
            "_fetch_feed",
            lambda url, headers: (_ for _ in ()).throw(
                ConnectionError("Network unreachable")
            ),
        )
        with pytest.raises(FetchError):
            list(adapter.fetch(reuters_source_config))

    def test_malformed_feed_raises_fetch_error(
        self, monkeypatch, reuters_source_config
    ):
        def bad_feed(url, headers):
            raise ValueError("Malformed XML")

        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", bad_feed)
        with pytest.raises(FetchError):
            list(adapter.fetch(reuters_source_config))

    @pytest.mark.integration
    @pytest.mark.skip(
        reason="Requires live network access — run manually with -m integration"
    )
    def test_live_reuters_rss_feed(self):
        config = make_source_config(
            url="https://feeds.reuters.com/reuters/topNews", tier=1
        )
        adapter = RssAdapter()
        results = list(adapter.fetch(config))
        assert len(results) > 0
        assert all("url" in item for item in results)
        assert all("published" in item for item in results)


# ---------------------------------------------------------------------------
# REST JSON adapter contract
# ---------------------------------------------------------------------------


class TestRestJsonAdapterContract:
    def test_rest_json_adapter_is_adapter_subclass(self):
        assert issubclass(RestJsonAdapter, Adapter)

    def test_fetch_yields_dicts(self, monkeypatch):
        raw = load_fixture("rest_json_sample.json")
        config = make_source_config(adapter="rest_json", tier=1)
        adapter = RestJsonAdapter()
        monkeypatch.setattr(adapter, "_fetch_json", lambda url, headers, params: [raw])

        results = list(adapter.fetch(config))
        assert len(results) >= 1
        assert all(isinstance(item, dict) for item in results)

    def test_each_item_has_required_fields(self, monkeypatch):
        raw = load_fixture("rest_json_sample.json")
        config = make_source_config(adapter="rest_json", tier=1)
        adapter = RestJsonAdapter()
        monkeypatch.setattr(adapter, "_fetch_json", lambda url, headers, params: [raw])

        for item in adapter.fetch(config):
            assert "url" in item
            assert "published" in item
            assert "raw_body" in item
            assert "raw_payload" in item

    def test_http_error_raises_fetch_error(self, monkeypatch):
        config = make_source_config(adapter="rest_json", tier=1)
        adapter = RestJsonAdapter()

        def bad_fetch(url, headers, params):
            raise ConnectionError("HTTP 503")

        monkeypatch.setattr(adapter, "_fetch_json", bad_fetch)
        with pytest.raises(FetchError):
            list(adapter.fetch(config))


# ---------------------------------------------------------------------------
# Conditional GET — 304 short-circuit and ETag/Last-Modified extraction
# (patches requests.get at the module level so no network is touched)
# ---------------------------------------------------------------------------


class TestRssConditionalGet:
    def test_304_raises_not_modified_signal(self, monkeypatch, reuters_source_config):
        """A 304 must surface as NotModifiedSignal, never an empty parsed feed."""
        resp = FakeResponse(status_code=304)
        monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

        adapter = RssAdapter()
        with pytest.raises(NotModifiedSignal):
            list(adapter.fetch(reuters_source_config))

    def test_200_attaches_etag_and_last_modified(
        self, monkeypatch, reuters_source_config
    ):
        resp = FakeResponse(
            status_code=200,
            headers={
                "ETag": '"v1-abc"',
                "Last-Modified": "Wed, 01 Jan 2026 00:00:00 GMT",
            },
            content=_ATOM_ONE_ENTRY,
        )
        monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

        items = list(RssAdapter().fetch(reuters_source_config))
        assert len(items) == 1
        assert items[0]["_etag"] == '"v1-abc"'
        assert items[0]["_last_modified"] == "Wed, 01 Jan 2026 00:00:00 GMT"

    def test_200_without_validators_still_works(
        self, monkeypatch, reuters_source_config
    ):
        """Conditional GET is opportunistic: no validator headers must not break fetch."""
        resp = FakeResponse(status_code=200, headers={}, content=_ATOM_ONE_ENTRY)
        monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

        items = list(RssAdapter().fetch(reuters_source_config))
        assert len(items) == 1
        assert "_etag" not in items[0]
        assert "_last_modified" not in items[0]


class TestRestJsonConditionalGet:
    def _config(self):
        return make_source_config(adapter="rest_json", tier=1)

    def test_304_raises_not_modified_signal(self, monkeypatch):
        resp = FakeResponse(status_code=304)
        monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

        with pytest.raises(NotModifiedSignal):
            list(RestJsonAdapter().fetch(self._config()))

    def test_200_attaches_validators(self, monkeypatch):
        resp = FakeResponse(
            status_code=200,
            headers={"ETag": '"json-1"'},
            content=b"[{}]",  # leading byte must look like JSON for the transport check
            json_data=[
                {
                    "url": "https://example.test/a/1",
                    "title": "T",
                    "content": "body",
                    "publishedAt": "2026-01-01T00:00:00Z",
                    "id": "1",
                }
            ],
        )
        monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

        items = list(RestJsonAdapter().fetch(self._config()))
        assert len(items) == 1
        assert items[0]["_etag"] == '"json-1"'
        assert "_last_modified" not in items[0]  # server sent only ETag


# ---------------------------------------------------------------------------
# Transport validation — fail-closed rejection of HTML challenge pages (Phase 2)
# ---------------------------------------------------------------------------


class TestRssTransportValidation:
    def test_html_challenge_page_at_200_is_rejected(
        self, monkeypatch, reuters_source_config
    ):
        """A 200 that returns an HTML challenge page must be refused, not parsed as a feed."""
        resp = FakeResponse(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=_CHALLENGE_HTML,
        )
        monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

        with pytest.raises(TransportError):
            list(RssAdapter().fetch(reuters_source_config))

    def test_transport_error_is_a_fetch_error(self):
        """Subclassing keeps the engine's per-source isolation working."""
        assert issubclass(TransportError, FetchError)


class TestRestJsonTransportValidation:
    def test_html_challenge_page_at_200_is_rejected(self, monkeypatch):
        resp = FakeResponse(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=_CHALLENGE_HTML,
        )
        monkeypatch.setattr(requests, "get", lambda *a, **k: resp)

        config = make_source_config(adapter="rest_json", tier=1)
        with pytest.raises(TransportError):
            list(RestJsonAdapter().fetch(config))


# ---------------------------------------------------------------------------
# REST-JSON adapter parse against a captured raw fixture (Phase 4)
# ---------------------------------------------------------------------------


class TestRestJsonRawParse:
    """Drive the real _fetch_json parse path over a raw JSON body, fully offline."""

    def _resp(self):
        import json

        raw_bytes = (
            Path(__file__).parent / "fixtures" / "rest_json_raw.json"
        ).read_bytes()
        return FakeResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=raw_bytes,  # transport check sniffs this
            json_data=json.loads(raw_bytes),  # .json() returns this
        )

    def test_parses_each_article_to_standard_shape(self, monkeypatch):
        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp())
        config = make_source_config(adapter="rest_json", tier=1)

        items = list(RestJsonAdapter().fetch(config))
        assert len(items) == 2
        for item in items:
            assert item["url"]
            assert item["published"]
            assert "raw_body" in item
            assert isinstance(item["raw_payload"], dict)
        assert items[0]["url"] == "https://news.example.com/markets/a1"

    def test_parsed_record_normalizes_to_valid_document(self, monkeypatch, fetched_at):
        from ingestion.pipeline.normalizer import normalize

        monkeypatch.setattr(requests, "get", lambda *a, **k: self._resp())
        config = make_source_config(adapter="rest_json", tier=1)

        first = next(iter(RestJsonAdapter().fetch(config)))
        doc = normalize(first, config, fetched_at)
        assert doc.url == "https://news.example.com/markets/a1"
        assert doc.title == "Markets rise on rate-cut hopes"
        assert "<p>" not in doc.body  # HTML stripped by the normalizer
        assert doc.published_date.tzinfo is not None
