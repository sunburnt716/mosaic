# Defines the SourceConfig schema — the typed representation of a single entry in sources.yaml.
#
# Each source in the registry is described entirely by config; no per-source code paths exist.
# This schema enforces what fields a source entry must carry and what values are legal.
#
# Required fields:
#   - name        : unique human-readable identifier (e.g. "reuters-rss", "sec-edgar")
#   - adapter     : string key that maps to a concrete Adapter class in adapters/registry.py
#                   (e.g. "rss", "rest_json", "edgar")
#   - tier        : integer trust level (0 = primary/regulatory, 1 = major wire, 2 = quality press,
#                   3 = signal/social); stamped on every Document at ingest, never changed later
#   - url         : the fetch endpoint — feed URL for RSS, API root for REST, or base URL for EDGAR
#
# Optional fields (adapter-specific, passed through as-is to the adapter):
#   - params      : dict of query parameters or adapter-specific knobs (pagination, filters, etc.)
#   - headers     : HTTP headers to include with requests (auth tokens, user-agent overrides)
#   - schedule    : cron expression or interval string controlling how often the source is polled
#   - enabled     : boolean flag; disabled sources are skipped by the engine without removing config
#
# Design rule: adding a new source means adding a YAML entry that validates against this schema,
# not writing new code. SourceConfig is the gatekeeper that enforces that contract.
