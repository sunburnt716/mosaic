"""
Contract tests for extraction/tickers.py.

Uses a small fake registry throughout — never the real (larger) extraction/config/
tickers.yaml — so these tests stay pinned to the matching contract, not to which
companies happen to be in the starter config.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extraction.tickers import extract_tickers, get_ticker_registry, load_ticker_registry

_REGISTRY = {
    "NVDA": ["Nvidia", "Nvidia Corporation"],
    "IT": [],  # deliberately alias-less: real ticker, also a common English word
    "AAPL": ["Apple", "Apple Inc"],
}


class TestExtractTickers:
    def test_matches_bare_ticker_symbol(self):
        assert extract_tickers("NVDA shares rose today.", _REGISTRY) == ["NVDA"]

    def test_matches_alias_case_insensitively(self):
        assert extract_tickers("nvidia reported earnings.", _REGISTRY) == ["NVDA"]

    def test_bare_ticker_match_is_case_sensitive(self):
        # "it" (lowercase) must never match the IT ticker — this is the whole reason
        # bare-symbol matching requires exact case.
        assert extract_tickers("it was a good quarter for the sector.", _REGISTRY) == []

    def test_bare_ticker_uppercase_does_match(self):
        assert extract_tickers("IT spending guidance was raised.", _REGISTRY) == ["IT"]

    def test_word_boundary_prevents_substring_match(self):
        # "NVDAX" is not NVDA — the word-boundary check must reject the substring.
        assert extract_tickers("NVDAX is an unrelated fund ticker.", _REGISTRY) == []

    def test_multiple_tickers_returned_sorted(self):
        result = extract_tickers("Apple and Nvidia both rallied.", _REGISTRY)
        assert result == ["AAPL", "NVDA"]

    def test_no_match_returns_empty_list(self):
        assert extract_tickers("Nothing relevant here.", _REGISTRY) == []

    def test_empty_registry_returns_empty_list(self):
        assert extract_tickers("Nvidia earnings beat.", {}) == []

    def test_duplicate_alias_hits_still_dedup_to_one_entry(self):
        text = "Nvidia, Nvidia Corporation, and NVDA all refer to the same company."
        assert extract_tickers(text, _REGISTRY) == ["NVDA"]


class TestLoadTickerRegistry:
    def test_missing_file_returns_empty_registry(self, tmp_path: Path):
        assert load_ticker_registry(tmp_path / "does-not-exist.yaml") == {}

    def test_loads_tickers_key_from_yaml(self, tmp_path: Path):
        config = tmp_path / "tickers.yaml"
        config.write_text("tickers:\n  NVDA: [\"Nvidia\"]\n", encoding="utf-8")
        assert load_ticker_registry(config) == {"NVDA": ["Nvidia"]}

    def test_empty_file_returns_empty_registry(self, tmp_path: Path):
        config = tmp_path / "tickers.yaml"
        config.write_text("", encoding="utf-8")
        assert load_ticker_registry(config) == {}


class TestGetTickerRegistry:
    def test_returns_injected_fake_without_reloading(self, monkeypatch):
        import extraction.tickers as tickers_module

        monkeypatch.setattr(tickers_module, "_registry", _REGISTRY)
        assert get_ticker_registry() is _REGISTRY

    def test_loads_real_default_config_when_cache_empty(self, monkeypatch):
        import extraction.tickers as tickers_module

        monkeypatch.setattr(tickers_module, "_registry", None)
        registry = get_ticker_registry()
        assert isinstance(registry, dict)
        assert "NVDA" in registry  # sanity check against the real starter config
