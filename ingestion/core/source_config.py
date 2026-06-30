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
    # (ingestion/pipeline/quality.py) ONLY where set; left as None, the gate falls back
    # to its source-agnostic module defaults. None of these ever drop records — they only
    # change when the gate emits a warning.
    #   max_fallback_title_rate: warn if the fraction of placeholder/empty titles exceeds
    #                            this.
    #   max_empty_body_rate:     warn if the fraction of empty bodies exceeds this
    #                            (and body is expected).
    #   min_records:             warn if a poll yields fewer than this many usable
    #                            (normalized) records — i.e. the batch the gate sees
    #                            post-validation.
    max_fallback_title_rate: float | None = None
    max_empty_body_rate: float | None = None
    min_records: int | None = None
