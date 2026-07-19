"""Contract tests for shared/tickers.py's local ticker-validation cache.

Offline: an injected fake `fetch_json` stands in for the real SEC company_tickers.json
response, pattern mirrors tests/test_body_enrichment.py's fake fetch injection.
"""

from __future__ import annotations

import shared.tickers as tickers_module
from shared.tickers import is_valid_ticker, refresh_tickers


def _fake_company_tickers() -> dict:
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    }


class _FakeFetcher:
    def __init__(self, response: dict):
        self._response = response
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, headers: dict) -> dict:
        self.calls.append((url, headers))
        return self._response


class TestRefreshTickers:
    def test_populates_cache_from_fetched_data(self, monkeypatch):
        monkeypatch.setattr(tickers_module, "_valid_tickers", set())
        fetch = _FakeFetcher(_fake_company_tickers())
        refresh_tickers("test@example.com", fetch_json=fetch)
        assert is_valid_ticker("AAPL")
        assert is_valid_ticker("NVDA")

    def test_sends_user_agent_header(self, monkeypatch):
        monkeypatch.setattr(tickers_module, "_valid_tickers", set())
        fetch = _FakeFetcher(_fake_company_tickers())
        refresh_tickers("test@example.com", fetch_json=fetch)
        url, headers = fetch.calls[0]
        assert url == tickers_module.SEC_TICKERS_URL
        assert headers["User-Agent"] == "test@example.com"

    def test_replaces_stale_cache_rather_than_merging(self, monkeypatch):
        monkeypatch.setattr(tickers_module, "_valid_tickers", {"STALE"})
        fetch = _FakeFetcher(_fake_company_tickers())
        refresh_tickers("test@example.com", fetch_json=fetch)
        assert not is_valid_ticker("STALE")


class TestIsValidTicker:
    def test_case_insensitive_lookup(self, monkeypatch):
        monkeypatch.setattr(tickers_module, "_valid_tickers", {"AAPL"})
        assert is_valid_ticker("aapl")
        assert is_valid_ticker("AAPL")
        assert is_valid_ticker("AaPl")

    def test_unknown_ticker_returns_false(self, monkeypatch):
        monkeypatch.setattr(tickers_module, "_valid_tickers", {"AAPL"})
        assert not is_valid_ticker("ZZZZ")

    def test_empty_cache_before_first_refresh(self, monkeypatch):
        monkeypatch.setattr(tickers_module, "_valid_tickers", set())
        assert not is_valid_ticker("AAPL")
