"""Generic RSS/Atom adapter — one implementation for all RSS/Atom sources.

Format-driven: per-source differences (URL, auth headers, schedule) live in
config/sources.json, not here. The HTTP fetch + feed parse is isolated in `_fetch_feed`
so the engine can wrap it (e.g. with conditional-GET validators) and tests can
substitute it. `fetch` only adds source-isolation error handling.
"""

from collections.abc import Iterable

from ingestion.adapters.base import (
    Adapter,
    FetchError,
    NotModifiedSignal,
    conditional_get_guard,
)
from ingestion.pipeline.validation import check_feed_well_formed, check_transport


class RssAdapter(Adapter):
    def fetch(self, config) -> Iterable[dict]:
        try:
            items = self._fetch_feed(config.url, config.headers)
        except (FetchError, NotModifiedSignal):
            raise
        except Exception as exc:  # network or parse failure -> isolate this source
            raise FetchError(f"RSS fetch failed for {config.url}: {exc}") from exc
        yield from items

    def _fetch_feed(self, url: str, headers: dict) -> list[dict]:
        """Fetch `url` and return one standard-shape dict per feed entry.

        Imports are local so the module loads without feedparser/requests installed
        (only live fetching needs them; unit tests substitute this method).
        """
        import feedparser
        import requests

        resp = requests.get(url, headers=headers or {}, timeout=30)
        # Conditional GET: raise NotModifiedSignal on 304, else capture ETag/Last-Modified.
        validators = conditional_get_guard(resp)
        resp.raise_for_status()
        # Fail-closed transport check: reject an HTML challenge page before we parse it
        # as a feed (a 200 text/html body is the classic feed-poisoning vector).
        check_transport(resp.content, resp.headers.get("Content-Type"), "xml")
        parsed = feedparser.parse(resp.content)
        # Fail-closed parse check: refuse a body that didn't parse as well-formed XML.
        check_feed_well_formed(parsed)
        return [
            {
                "url": entry.get("link"),
                "title": entry.get("title"),
                "raw_body": entry.get("summary") or entry.get("description") or "",
                "published": entry.get("published") or entry.get("updated"),
                "source_article_id": entry.get("id")
                or entry.get("guid")
                or entry.get("link"),
                "raw_payload": dict(entry),
                # Validators ride along on each item; the engine pops them (last wins)
                # and persists them to poll_state for the next conditional GET.
                **validators,
            }
            for entry in parsed.entries
        ]
