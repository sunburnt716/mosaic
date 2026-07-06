"""Pure transformation stage: raw adapter dict + SourceConfig -> validated Document.

The normalizer is the boundary between the adapter world (format-specific, messy)
and the pipeline world (typed, validated, canonical). It is the only place that
produces Documents from raw input. See the contract in test_normalizer.py.

Every adapter (rss, rest_json) already yields the standard shape (url, title, raw_body,
published, source_article_id, raw_payload) regardless of source — that is the point of
being format-driven. Field-name mappings therefore default to that standard shape and
are overridden per-source only via config.field_mappings (e.g.
field_mappings={"body": "raw_text"}), for the rare source whose entries don't fit.
doc_type is stamped verbatim from config.doc_type (the validated top-level field on
SourceConfig) — never read from content.

This function is pure: same input always yields the same output, no I/O, no state.
"""

import html
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

from ingestion.core.document import Document
from ingestion.core.source_config import SourceConfig
from ingestion.pipeline.hashing import content_hash, document_id, identity_key
from ingestion.pipeline.transforms import get_transform

_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_CLOSE_RE = re.compile(r"(?i)</(p|div|li|h[1-6]|section|article)>")
_BR_RE = re.compile(r"(?i)<br\s*/?>")

# Standard adapter-output key for each Document field, overridable via field_mappings.
_DEFAULT_FIELD_KEYS = {
    "url": "url",
    "title": "title",
    "body": "raw_body",
    "published_date": "published",
    "source_article_id": "source_article_id",
}


class NormalizationError(Exception):
    """Raised when a raw item cannot be turned into a valid Document.

    Covers missing/unparseable required fields (url, published_date, source_name).
    """


def _clean_html(raw_html: str | None) -> str:
    """Strip HTML to clean plain text, preserving paragraph boundaries as newlines."""
    if not raw_html:
        return ""
    text = _BLOCK_CLOSE_RE.sub("\n", raw_html)
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _parse_date(value) -> datetime:
    """Coerce a raw timestamp (ISO-8601 or RFC-2822) to a tz-aware UTC datetime."""
    if not value or not isinstance(value, str):
        raise NormalizationError(f"missing or non-string published date: {value!r}")

    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = None

    if parsed is None:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            parsed = None

    if parsed is None:
        raise NormalizationError(f"unparseable published date: {value!r}")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _field(raw: dict, config: SourceConfig, field_name: str):
    """Look up `field_name` in `raw` using the source's override, else the standard key."""
    key = (config.field_mappings or {}).get(field_name, _DEFAULT_FIELD_KEYS[field_name])
    return raw.get(key)


def normalize(raw: dict, config: SourceConfig, fetched_at: datetime) -> Document:
    # Apply per-source transform before generic field-mapping, if one is registered.
    if config.transform:
        raw = get_transform(config.transform)(raw, config)

    url = _field(raw, config, "url")
    if not url:
        raise NormalizationError(f"missing url for source {config.name!r}")
    # Per-record contract: reject any URL that isn't an absolute http(s) URL (no scheme
    # or no host). A citation URL that synthesis cannot resolve is worthless downstream.
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        raise NormalizationError(f"unparseable url for source {config.name!r}: {url!r}")

    published_date = _parse_date(_field(raw, config, "published_date"))

    title = _field(raw, config, "title") or ""
    body = _clean_html(_field(raw, config, "body"))
    article_id = _field(raw, config, "source_article_id") or url
    raw_payload = raw.get("raw_payload", raw)

    chash = content_hash(body)
    ikey = identity_key(config.name, article_id)

    return Document(
        id=document_id(ikey, chash),
        content_hash=chash,
        identity_key=ikey,
        source_name=config.name,
        url=url,
        tier=config.tier,  # stamped from config — never read from content
        published_date=published_date,
        title=title,
        body=body,
        doc_type=config.doc_type,  # advisory hint, stamped from config — never inferred here
        raw_payload=raw_payload,
        fetched_at=fetched_at,
        # tickers/sectors/key_points, status, document_type, validation_warnings are left
        # to their defaults — later stages (enrichment, processing) populate them.
    )
