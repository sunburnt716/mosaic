"""The Adapter abstract base class and shared adapter exceptions.

Adapter.fetch(config) -> Iterable[dict]
  Yields one dict per logical document. Each dict must contain:
  url, published (raw timestamp), raw_body, raw_payload.
  Raises FetchError (never bare Exception) on network or parse failure.

Conditional-GET contract (every HTTP adapter must honour it):
  - The engine merges If-None-Match / If-Modified-Since headers into config.headers
    before calling fetch (built from the previously stored validators in poll_state).
  - On HTTP 304 Not Modified, the adapter MUST raise NotModifiedSignal (do not parse an
    empty body into zero items — that would silently look like an empty feed).
  - On HTTP 200, the adapter SHOULD attach the response's ETag / Last-Modified to each
    yielded dict as `_etag` / `_last_modified`. The engine pops these and persists them
    to poll_state so the next poll can send them back. Conditional GET is opportunistic:
    a source that sends no validators still works (the engine simply stores None).

FetchError        — network / parse failure; engine isolates the source and continues.
TransportError    — fail-closed structural validation failure (empty body, wrong format,
                    malformed feed). A FetchError subclass so source isolation still applies,
                    but surfaced distinctly so the run summary can flag a refused batch.
NotModifiedSignal — server returned 304; engine short-circuits without parsing.
ConfigError       — unknown adapter key or malformed source config; fatal at startup.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable


class FetchError(Exception):
    """Raised by an adapter on network or parse failure during fetch()."""


class TransportError(FetchError):
    """Fail-closed structural validation failure at the fetch boundary.

    Raised when a 200 response is structurally unusable — empty body, an HTML challenge
    page where a feed/JSON was expected, or a body that does not parse as its declared
    format. A subclass of FetchError so the engine's per-source isolation already applies;
    the engine catches it explicitly (before FetchError) to flag the batch as refused.
    A known-broken payload must never reach normalize/dedup/store — hence fail-closed.
    """


class NotModifiedSignal(Exception):
    """Raised when the server returns 304 Not Modified.

    Not a failure — engine catches this as a clean short-circuit: update last_polled_at,
    skip parse/normalize/dedup/store, move to next source.
    """


class ConfigError(Exception):
    """Raised for unknown adapter keys or malformed source config entries."""


def conditional_get_guard(resp) -> dict:
    """Apply the shared conditional-GET contract to an HTTP response.

    Raises NotModifiedSignal on a 304 so the engine short-circuits instead of parsing
    an empty body into zero items. Otherwise returns the response's validators as a dict
    ({"_etag": ..., "_last_modified": ...}, only for headers the server actually sent),
    ready to be merged into each yielded item for the engine to persist to poll_state.

    Used by every HTTP adapter so 304 handling and validator extraction stay identical.
    """
    if resp.status_code == 304:
        raise NotModifiedSignal(f"304 Not Modified for {getattr(resp, 'url', '<url>')}")

    validators: dict = {}
    etag = resp.headers.get("ETag")
    last_modified = resp.headers.get("Last-Modified")
    if etag:
        validators["_etag"] = etag
    if last_modified:
        validators["_last_modified"] = last_modified
    return validators


class Adapter(ABC):
    @abstractmethod
    def fetch(self, config) -> Iterable[dict]:
        """Fetch from the source described by `config` and yield raw item dicts."""
        raise NotImplementedError
