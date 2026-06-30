<<<<<<< HEAD
"""
SourceConfig — the typed, validated representation of a single source registry entry.

This is the gatekeeper for the config-over-code principle: every source is described
entirely by a SourceConfig, and adding a new source means adding a YAML entry, not
writing new code paths.

Per-source config contract (all fields required unless noted):
  url, tier, auth, field_mappings  — the four fields every source must supply.
  poll_interval                    — how often the scheduler dispatches this source.
  adapter                          — which format adapter handles this source.
  doc_type                         — "article" or "filing"; controls downstream chunking.

Validation is strict and runs at startup (load_sources). A source that fails
validation kills the process before any network calls are made — partial configs
must never reach production.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys that map to registered Adapter classes in adapters/registry.py.
# Update this set when a new adapter is added there.
_VALID_ADAPTERS: frozenset[str] = frozenset({"rss", "rest_json", "edgar"})

# Valid values for doc_type, which controls which chunking strategy is applied
# downstream: filings are chunked by section, articles by paragraph.
_VALID_DOC_TYPES: frozenset[str] = frozenset({"article", "filing"})

# Tier boundaries. Tier is stamped at ingest and must never be inferred later.
_TIER_MIN = 0
_TIER_MAX = 3

# Regex for human-readable duration strings: "5m", "1h", "1h30m", "2h15m30s".
# All three groups are optional, but at least one must be present.
_INTERVAL_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------


def _parse_interval(raw: str) -> timedelta:
    """Parse a human-readable duration string into a timedelta.

    Accepted formats (case-sensitive, no spaces):
      "30s", "5m", "1h", "1h30m", "2h15m30s"

    Raises ValueError on unrecognised format, all-zero value, or empty string.
    """
    m = _INTERVAL_RE.fullmatch(raw.strip())
    # The regex matches the empty string (all groups None), so check that too.
    if not m or not any(m.groups()):
        raise ValueError(
            f"Invalid poll_interval {raw!r}. "
            "Use a combination of hours/minutes/seconds, e.g. '5m', '1h', '1h30m'."
        )
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    if delta.total_seconds() <= 0:
        # e.g. "0h0m0s" passes the regex but is nonsensical for a poll interval.
        raise ValueError(
            f"poll_interval must be positive; {raw!r} resolves to zero seconds."
        )
    return delta


# ---------------------------------------------------------------------------
# SourceConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceConfig:
    """Immutable, validated representation of one source registry entry.

    Constructed only by load_sources() — never build directly from user input.
    All fields satisfy the per-source config contract on construction.

    Field notes:
      tier         — stamped on every Document produced by this source at ingest.
                     Never inferred from content downstream. 0 = most trusted.
      field_mappings — maps schema field names to adapter-specific payload keys,
                     e.g. {"body": "summary", "published_date": "published"}.
                     The normalizer reads this to translate adapter output without
                     per-source code branches.
      auth         — adapter-specific credentials, e.g. {"token": "Bearer xyz"}.
                     Not stored in sources.yaml in plain text for production; the
                     loader accepts it here so tests and dev configs can supply it.
      doc_type     — controls chunking: "article" → by paragraph,
                     "filing" → by section.
      params       — catch-all for adapter-specific knobs not covered above
                     (e.g. pagination config, result filters).
      headers      — HTTP headers added to every request for this source.
    """

    name: str
    adapter: str
    tier: int
    url: str
    poll_interval: timedelta
    doc_type: Literal["article", "filing"]
    field_mappings: dict[str, str]
    auth: dict[str, Any]
    enabled: bool
    params: dict[str, Any]
    headers: dict[str, str]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require(entry: dict[str, Any], key: str, source_label: str) -> Any:
    """Return entry[key] or raise ValueError naming the source and the missing key."""
    if key not in entry:
        raise ValueError(f"{source_label}: missing required field '{key}'.")
    return entry[key]


def _validate_entry(entry: dict[str, Any], index: int) -> SourceConfig:
    """Validate one raw YAML mapping and return a SourceConfig.

    Raises ValueError with a message that names the source (or its index if
    the name is missing/invalid) and describes exactly what is wrong.
    """
    # Resolve the label we'll use in all error messages for this entry.
    raw_name = entry.get("name", f"<index {index}>")
    label = f"Source {raw_name!r}"

    # --- name ---
    name = _require(entry, "name", label)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label}: 'name' must be a non-empty string.")

    # --- adapter ---
    adapter = _require(entry, "adapter", label)
    if adapter not in _VALID_ADAPTERS:
        raise ValueError(
            f"{label}: adapter {adapter!r} is not registered. "
            f"Valid adapters: {sorted(_VALID_ADAPTERS)}."
        )

    # --- tier ---
    tier = _require(entry, "tier", label)
    if not isinstance(tier, int) or isinstance(tier, bool):
        # YAML parses `tier: true` as bool True; reject that explicitly.
        raise ValueError(
            f"{label}: 'tier' must be an integer, got {tier!r} ({type(tier).__name__})."
        )
    if not (_TIER_MIN <= tier <= _TIER_MAX):
        raise ValueError(
            f"{label}: 'tier' must be {_TIER_MIN}–{_TIER_MAX}, got {tier}."
        )

    # --- url ---
    url = _require(entry, "url", label)
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{label}: 'url' must be a non-empty string.")

    # --- poll_interval ---
    raw_interval = _require(entry, "poll_interval", label)
    if not isinstance(raw_interval, str):
        raise ValueError(
            f"{label}: 'poll_interval' must be a string like '5m', got {raw_interval!r}."
        )
    # _parse_interval raises ValueError on bad format or zero duration.
    poll_interval = _parse_interval(raw_interval)

    # --- doc_type (optional, defaults to "article") ---
    raw_doc_type = entry.get("doc_type", "article")
    if raw_doc_type not in _VALID_DOC_TYPES:
        raise ValueError(
            f"{label}: 'doc_type' must be one of {sorted(_VALID_DOC_TYPES)}, "
            f"got {raw_doc_type!r}."
        )
    doc_type: Literal["article", "filing"] = raw_doc_type  # type: ignore[assignment]

    # --- field_mappings (optional, defaults to {}) ---
    field_mappings = entry.get("field_mappings", {})
    if not isinstance(field_mappings, dict):
        raise ValueError(f"{label}: 'field_mappings' must be a mapping.")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in field_mappings.items()):
        raise ValueError(f"{label}: 'field_mappings' must be a str→str mapping.")

    # --- auth (optional, defaults to {}) ---
    auth = entry.get("auth", {})
    if not isinstance(auth, dict):
        raise ValueError(f"{label}: 'auth' must be a mapping.")

    # --- enabled (optional, defaults to True) ---
    enabled = entry.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"{label}: 'enabled' must be a boolean (true/false), got {enabled!r}."
        )

    # --- params (optional, defaults to {}) ---
    params = entry.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"{label}: 'params' must be a mapping.")

    # --- headers (optional, defaults to {}) ---
    headers = entry.get("headers", {})
    if not isinstance(headers, dict):
        raise ValueError(f"{label}: 'headers' must be a mapping.")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
        raise ValueError(f"{label}: 'headers' must be a str→str mapping.")

    return SourceConfig(
        name=str(name).strip(),
        adapter=adapter,
        tier=tier,
        url=str(url).strip(),
        poll_interval=poll_interval,
        doc_type=doc_type,
        field_mappings=dict(field_mappings),
        auth=dict(auth),
        enabled=enabled,
        params=dict(params),
        headers=dict(headers),
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_sources(config_path: Path) -> list[SourceConfig]:
    """Parse sources.yaml and validate every entry into a typed SourceConfig.

    Fails fast: the very first validation error raises, so the process dies at
    startup rather than discovering a broken source at its first poll time.

    Validates at startup (this function):
      - required fields are present and of the correct type
      - adapter is a registered key
      - tier is in range 0–3
      - poll_interval is a positive, parseable duration
      - doc_type is "article" or "filing"
      - source names are unique (duplicate names corrupt identity_key in L2 dedup)

    Deferred to the normalizer (per-article, at runtime):
      - whether the fetched payload contains the field_mappings keys
      - whether published_date is parseable
      - whether body is non-empty after HTML stripping

    Raises:
        FileNotFoundError: config_path does not exist.
        ValueError: any entry fails validation, or the file structure is wrong.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Source registry not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())

    if not isinstance(raw, dict) or "sources" not in raw:
        raise ValueError(
            f"{config_path}: expected a top-level mapping with a 'sources' key."
        )

    entries = raw["sources"]
    if not isinstance(entries, list) or len(entries) == 0:
        raise ValueError(
            f"{config_path}: 'sources' must be a non-empty list of source mappings."
        )

    sources: list[SourceConfig] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{config_path}: entry at index {i} is not a mapping (got {type(entry).__name__})."
            )
        sources.append(_validate_entry(entry, i))

    # Enforce unique names. Duplicate names would cause identity_key collisions
    # across sources, silently corrupting L2 dedup (same-article-updated detection).
    seen_names: set[str] = set()
    for source in sources:
        if source.name in seen_names:
            raise ValueError(
                f"{config_path}: duplicate source name {source.name!r}. "
                "Source names must be unique — duplicates corrupt L2 dedup identity keys."
            )
        seen_names.add(source.name)

    return sources
=======
"""The SourceConfig schema — the typed representation of one entry in config/sources.json.

Each source is described entirely by config; no per-source code paths exist.
Adding a source means adding a YAML entry that validates against this schema,
not writing new code.

  Required: name, adapter, tier, url
  Optional: enabled, params (adapter-specific knobs incl. field mappings + doc_type),
            headers (auth / user-agent), poll_interval (scheduler cadence),
            transform (name of a registered per-source transform function),
            expects (quality gate hints: which fields must be non-empty),
            max_fallback_title_rate / max_empty_body_rate / min_records
            (optional per-source quality-gate thresholds; see below)
"""

from dataclasses import dataclass, field


@dataclass
class SourceConfig:
    name: str
    adapter: str  # key into adapters/registry.py: "rss" | "rest_json"
    tier: int  # 0 primary/regulatory · 1 wire · 2 quality press · 3 signal/social
    url: str
    enabled: bool = True
    params: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)
    poll_interval: str | None = None  # consumed by the scheduler; see caching strategy
    # Optional per-source transform applied before generic field-mapping in the normalizer.
    # Value is a name registered in ingestion/pipeline/transforms.py.
    transform: str | None = None
    # Quality gate hints: which output fields must be non-empty for this source.
    # Used by ingestion/pipeline/quality.py to detect silent parse failures.
    # Example: {"title": True, "url": True, "body": False}
    expects: dict = field(default_factory=dict)
    # Optional per-source quality-gate thresholds. Each tunes the soft batch gate
    # (ingestion/pipeline/quality.py) ONLY where set; left as None, the gate falls back to
    # its source-agnostic module defaults. None of these ever drop records — they only
    # change when the gate emits a warning.
    #   max_fallback_title_rate: warn if the fraction of placeholder/empty titles exceeds this.
    #   max_empty_body_rate:     warn if the fraction of empty bodies exceeds this (and body is expected).
    #   min_records:             warn if a poll yields fewer than this many usable (normalized)
    #                            records — i.e. the batch the gate actually sees post-validation.
    max_fallback_title_rate: float | None = None
    max_empty_body_rate: float | None = None
    min_records: int | None = None
>>>>>>> main
