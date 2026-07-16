"""
Ticker enrichment — Document.tickers, populated by symbol-list matching over title+body.

Runs alongside Phase 0 type inference: both are content-derived enrichment that happens
before chunking, so a Chunk built downstream can carry `ticker` (see chunk.py) and
Chroma can filter on it (see chroma_store.py, retrieval/search.py's build_where_clause).

Matching is plain code, no model call — same "prefer plain code + embeddings before
reaching for an LLM" preference CLAUDE.md states for routing/dedup. Two match types per
registry entry:
  - the bare ticker symbol itself, matched case-sensitively at a word boundary. Case
    sensitivity matters: several real tickers ("IT", "ON", "ALL", "ARE") are also common
    English words, so a lowercase/mixed-case occurrence must not count as a match.
  - any alias (company name / variant), matched case-insensitively at a word boundary.

The registry is config, not code (`extraction/config/tickers.yaml`), same convention as
`ingestion/config/sources.yaml` — extending ticker coverage never needs a code change.
Loaded lazily and cached at module level, mirroring `extraction/utils/embedding.py` and
`extraction/utils/tokenization.py`'s lazy-load pattern; the offline test suite injects a
small fake registry rather than loading the real (larger) one.

Exports:
  TickerRegistry              — dict[str, list[str]], ticker -> aliases
  load_ticker_registry(path)  — parse one YAML file into a TickerRegistry
  get_ticker_registry()       — cached singleton, loaded from the default config path
  extract_tickers(text, reg)  — match text against a registry, sorted matched tickers
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

TickerRegistry = dict[str, list[str]]

_DEFAULT_REGISTRY_PATH = Path(__file__).parent / "config" / "tickers.yaml"

# Lazily populated cache. None until first real use; tests overwrite it with a small
# fake registry so the offline suite never depends on the real config file's contents.
_registry: TickerRegistry | None = None


def load_ticker_registry(path: Path | None = None) -> TickerRegistry:
    """Parse a ticker registry YAML file. A missing file yields an empty registry."""
    path = path or _DEFAULT_REGISTRY_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("tickers", {}) or {}


def get_ticker_registry() -> TickerRegistry:
    """Return the cached ticker registry, loading it from the default config path once."""
    global _registry
    if _registry is None:
        _registry = load_ticker_registry()
    return _registry


def _word_boundary_pattern(term: str, *, case_sensitive: bool) -> re.Pattern:
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", flags)


def extract_tickers(text: str, registry: TickerRegistry) -> list[str]:
    """Match `text` against `registry`, returning sorted, deduplicated tickers found.

    A ticker matches if its bare symbol appears case-sensitively (word-boundary), or any
    of its aliases appears case-insensitively (word-boundary). Returns [] for no matches
    or an empty registry — this is advisory enrichment, never a hard requirement.
    """
    matched: set[str] = set()
    for ticker, aliases in registry.items():
        if _word_boundary_pattern(ticker, case_sensitive=True).search(text):
            matched.add(ticker)
            continue
        if any(_word_boundary_pattern(alias, case_sensitive=False).search(text) for alias in aliases):
            matched.add(ticker)
    return sorted(matched)
