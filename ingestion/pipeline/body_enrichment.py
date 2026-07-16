"""Body enrichment — fetch a source's real page text when the feed only carries a snippet.

Some feeds (SEC EDGAR's getcurrent Atom feed) hand the adapter only a headline/index-page
`summary`, not the document body. For sources that opt in via `SourceConfig.body_fetch`,
this stage fetches the real content at ingest time and replaces `raw_body` BEFORE the
record reaches `normalize()` — which matters because `normalize` derives `content_hash` and
`document_id` (and thus downstream `chunk_id`) from the body. Enriching here means the real
body lands in the raw store, so extraction stays offline-replayable.

Contract, mirroring `pipeline/transforms.py`:
  - Strategies register with `@register("name")` and are referenced from sources.yaml via
    `body_fetch:`. `enrich_body` dispatches on `config.body_fetch`.
  - **Best-effort: never raises, never drops.** Any fetch/parse failure logs a warning and
    returns the record unchanged (keeping the feed's `summary` body). A missing body is a
    quality signal, never a lost record.
  - Network lives behind an injected `fetch_url` callable (default: a lazy-`requests` GET),
    so the offline test suite substitutes fixtures — same pattern as `RssAdapter._fetch_feed`.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlparse

from ingestion.pipeline.html_text import clean_html

if TYPE_CHECKING:
    from ingestion.core.source_config import SourceConfig

_log = logging.getLogger(__name__)

# (url, headers) -> page text. Injected so tests never hit the network.
FetchUrl = Callable[[str, dict], str]
# (raw_entry, config, fetch_url) -> cleaned body text ("" if it can't produce one).
Strategy = Callable[[dict, "SourceConfig", FetchUrl], str]

# Cap enriched bodies so a pathological exhibit can't blow up embedding cost / Chroma size.
DEFAULT_MAX_BODY_CHARS = 50_000

_DEFAULT_TIMEOUT_SECONDS = 30
# SEC asks for <= 10 requests/second; pace real requests well under that. The default
# fetcher throttles itself so both the index and primary-document fetches stay polite,
# while injected test fetchers (which don't call this) never sleep.
_REQUEST_THROTTLE_SECONDS = 0.2

_REGISTRY: dict[str, Strategy] = {}


def register(name: str) -> Callable[[Strategy], Strategy]:
    """Decorator registering a body-fetch strategy under `name`."""

    def decorator(fn: Strategy) -> Strategy:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_strategy(name: str) -> Strategy:
    """Return the strategy for `name`, or raise ValueError if unknown."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown body_fetch strategy {name!r}. Registered: {sorted(_REGISTRY)}.")
    return _REGISTRY[name]


def default_fetch_url(url: str, headers: dict) -> str:
    """Fetch `url` and return its decoded text. Lazy-imports requests (only live runs need it)."""
    import requests

    time.sleep(_REQUEST_THROTTLE_SECONDS)  # polite pacing before each real request
    resp = requests.get(url, headers=headers or {}, timeout=_DEFAULT_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.text


def _body_key(config: "SourceConfig") -> str:
    """The raw-dict key `normalize` reads the body from (default 'raw_body', overridable)."""
    return (config.field_mappings or {}).get("body", "raw_body")


def enrich_body(
    raw: dict, config: "SourceConfig", *, fetch_url: FetchUrl = default_fetch_url
) -> dict:
    """Return `raw` with its body replaced by fetched full text, per `config.body_fetch`.

    Returns `raw` unchanged when the source opts out (`body_fetch is None`) or when
    enrichment can't produce a non-empty body (fetch/parse failure, empty document) —
    best-effort, never raising.
    """
    strategy_name = config.body_fetch
    if not strategy_name:
        return raw

    try:
        strategy = get_strategy(strategy_name)
    except ValueError as exc:
        _log.warning("body enrichment misconfigured for %s: %s", config.name, exc)
        return raw

    try:
        body = strategy(raw, config, fetch_url)
    except Exception as exc:  # noqa: BLE001 — any failure falls back to the feed summary
        _log.warning(
            "body enrichment failed for %s (%s): %s — keeping feed summary",
            config.name,
            raw.get("url"),
            exc,
        )
        return raw

    if not body or not body.strip():
        return raw

    enriched = dict(raw)
    enriched[_body_key(config)] = body[:DEFAULT_MAX_BODY_CHARS]
    return enriched


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# href="..." on an EDGAR filing index page; captures the link target verbatim.
_HREF_RE = re.compile(r'href\s*=\s*"([^"]+)"', re.IGNORECASE)
_DOC_SUFFIXES = (".htm", ".html", ".txt")


def _accession_folder(index_url: str) -> str:
    """The archive folder path the filing's documents live under, e.g.
    '/Archives/edgar/data/1083446/000110465926082892/'."""
    path = urlparse(index_url).path
    return path.rsplit("/", 1)[0] + "/"


def _resolve_primary_document(index_html: str, index_url: str) -> str | None:
    """Find the primary document URL on a filing index page.

    The index page lists its documents in order (Document Format Files first), each as a
    link into the same accession folder. The primary document is the first such link that
    isn't the index page itself — a dependency-free heuristic that matches SEC's layout.
    """
    folder = _accession_folder(index_url)
    index_path = urlparse(index_url).path
    for href in _HREF_RE.findall(index_html):
        abs_url = urljoin(index_url, href)
        path = urlparse(abs_url).path
        if not path.startswith(folder):
            continue  # link out of this filing's folder (nav, other filings)
        if path == index_path or path.lower().endswith(("-index.htm", "-index.html")):
            continue  # the index page itself
        if path.lower().endswith(_DOC_SUFFIXES):
            return abs_url
    return None


@register("edgar_filing")
def edgar_filing(raw: dict, config: "SourceConfig", fetch_url: FetchUrl) -> str:
    """Two-hop EDGAR fetch: filing index page -> primary document -> cleaned text.

    `raw['url']` is the '…-index.htm' page; the actual 8-K body is a document linked from
    it. Returns '' if the index has no resolvable primary document (caller keeps the
    summary).
    """
    index_url = raw.get("url")
    if not index_url:
        return ""
    index_html = fetch_url(index_url, config.headers)
    primary_url = _resolve_primary_document(index_html, index_url)
    if not primary_url:
        _log.warning("edgar_filing: no primary document found on index %s", index_url)
        return ""
    document_html = fetch_url(primary_url, config.headers)
    return clean_html(document_html)
