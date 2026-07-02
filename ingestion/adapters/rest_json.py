"""Generic REST-JSON adapter — one implementation for all JSON API sources.

Format-driven: endpoint, auth, pagination, and response field names are expressed
through SourceConfig params, not branching code. The HTTP fetch + JSON traversal is
isolated in `_fetch_json` so the engine can wrap it and tests can substitute it.
`fetch` only adds source-isolation error handling.
"""

from collections.abc import Iterable

from ingestion.adapters.base import (
    Adapter,
    FetchError,
    NotModifiedSignal,
    conditional_get_guard,
)
from ingestion.pipeline.validation import check_transport


class RestJsonAdapter(Adapter):
    def fetch(self, config) -> Iterable[dict]:
        try:
            items = self._fetch_json(config.url, config.headers, config.params)
        except (FetchError, NotModifiedSignal):  # 304 must pass through, not be wrapped
            raise
        except Exception as exc:  # non-2xx / parse failure -> isolate this source
            raise FetchError(f"REST fetch failed for {config.url}: {exc}") from exc
        yield from items

    def _fetch_json(self, url: str, headers: dict, params: dict) -> list[dict]:
        """GET `url` and return one standard-shape dict per item in the response.

        `params["items_path"]` (dotted) locates the array of items in the response;
        absent, the top-level response is assumed to be the list. Field names map via
        params with sensible defaults. Import is local so the module loads without
        requests installed (only live fetching needs it; unit tests substitute this).
        """
        import requests

        query = {k: v for k, v in (params or {}).items() if not k.endswith("_field")}
        query.pop("items_path", None)
        query.pop("doc_type", None)

        resp = requests.get(url, headers=headers or {}, params=query, timeout=30)
        # Conditional GET: raise NotModifiedSignal on 304, else capture ETag/Last-Modified.
        validators = conditional_get_guard(resp)
        resp.raise_for_status()
        # Fail-closed transport check: reject an HTML challenge page before parsing JSON.
        check_transport(resp.content, resp.headers.get("Content-Type"), "json")
        data = resp.json()

        items = data
        items_path = (params or {}).get("items_path")
        if items_path:
            for key in items_path.split("."):
                items = items[key]

        url_field = (params or {}).get("url_field", "url")
        title_field = (params or {}).get("title_field", "title")
        body_field = (params or {}).get("body_field", "content")
        published_field = (params or {}).get("published_field", "publishedAt")
        id_field = (params or {}).get("id_field", "id")

        return [
            {
                "url": item.get(url_field),
                "title": item.get(title_field),
                "raw_body": item.get(body_field) or "",
                "published": item.get(published_field),
                "source_article_id": item.get(id_field) or item.get(url_field),
                "raw_payload": item,
                # Validators ride along on each item; the engine pops them and persists
                # them to poll_state for the next conditional GET.
                **validators,
            }
            for item in items
        ]
