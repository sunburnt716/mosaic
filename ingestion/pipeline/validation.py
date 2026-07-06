"""Fail-closed structural validation layers, applied at the fetch boundary before normalize.

A known-broken payload must never reach the store, so these checks *refuse* (raise
TransportError) rather than drop-and-continue. They are pure and source-agnostic — they
judge the transport/format, never the source — so any adapter can call them.

Two ordered layers (the per-record contract is the third layer and lives in the
normalizer, which drops-and-counts bad records instead of refusing the whole batch):

  1. check_transport(body, content_type, expected_format)
     The response is non-empty and is not an HTML challenge/error page where a feed or
     JSON was expected. This is how a Cloudflare/anti-bot page (HTTP 200, text/html)
     silently poisons a feed; we reject it here.

  2. check_feed_well_formed(parsed)
     A feed body actually parsed as well-formed XML. feedparser is lenient and sets
     `bozo` for benign reasons, so we only refuse when it flagged the feed malformed AND
     extracted zero entries — i.e. it genuinely could not read the feed.
"""

from ingestion.adapters.base import TransportError

# How much of the body to sniff. The discriminating markers (<!doctype html, <?xml, {, [)
# are always at the very start, so a small window is plenty.
_SNIFF_BYTES = 512
_BOM = b"\xef\xbb\xbf"


def _leading(body: bytes) -> bytes:
    """Return the lowercased leading bytes with whitespace and a UTF-8 BOM stripped."""
    stripped = body.lstrip()
    if stripped[:3] == _BOM:
        stripped = stripped[3:].lstrip()
    return stripped[:_SNIFF_BYTES].lower()


def check_transport(body: bytes, content_type: str | None, expected_format: str) -> None:
    """Refuse an empty body or an HTML page where `expected_format` (xml|json) was wanted.

    Body-sniffing is primary; content-type is a secondary signal because servers mislabel
    it routinely (a valid feed served as text/html is common; a challenge page is not).
    """
    if not body or not body.strip():
        raise TransportError(f"empty response body (expected {expected_format})")

    head = _leading(body)
    ct = (content_type or "").lower()
    looks_html = head.startswith(b"<!doctype html") or head.startswith(b"<html")

    if expected_format == "xml":
        # A feed starts with <?xml, <rss, or <feed; a challenge page is an HTML document.
        feed_marker = head.startswith((b"<?xml", b"<rss", b"<feed"))
        if looks_html or ("text/html" in ct and not feed_marker):
            raise TransportError(
                "expected an XML/Atom feed but got an HTML page "
                f"(content-type={content_type!r}); likely a challenge/error page"
            )
    elif expected_format == "json":
        # JSON starts with { or [; anything starting with markup is not JSON.
        if head.startswith(b"<") or "text/html" in ct:
            raise TransportError(
                "expected JSON but got markup/HTML "
                f"(content-type={content_type!r}); likely a challenge/error page"
            )
    else:  # pragma: no cover - guards against a typo'd expected_format at call sites
        raise ValueError(f"unknown expected_format {expected_format!r}")


def check_feed_well_formed(parsed) -> None:
    """Refuse a feed body that did not parse as well-formed XML.

    Only fail-closed when feedparser both flagged the feed malformed (`bozo`) AND produced
    zero entries — a well-formed-but-empty feed (bozo=0, entries=[]) is left for the soft
    quality gate, not refused here.
    """
    if getattr(parsed, "bozo", 0) and not getattr(parsed, "entries", None):
        exc = getattr(parsed, "bozo_exception", None)
        raise TransportError(f"feed did not parse as well-formed XML: {exc!r}")
