# Pure transformation stage: maps a raw adapter dict + SourceConfig into a validated Document.
#
# The normalizer is the boundary between the adapter world (format-specific, messy, variable)
# and the pipeline world (typed, validated, canonical). Everything downstream depends on Documents;
# the normalizer is the only place that produces them from raw input.
#
# Responsibilities:
#   normalize(raw: dict, config: SourceConfig, fetched_at: datetime) -> Document
#     - Extract and coerce each Document field from the raw dict using field-name mappings.
#       Field-name mappings (e.g. which key holds the body, which holds the published date)
#       come from SourceConfig.params so no per-source code branches are needed here.
#     - Coerce published_date to a timezone-aware UTC datetime; raise NormalizationError if
#       the raw timestamp is absent or unparseable — missing dates break recency ranking.
#     - Strip HTML tags from body to produce clean plain text for the chunker.
#     - Stamp tier from config.tier — never read it from content.
#     - Set doc_type based on config (e.g. "filing" for EDGAR, "article" for everything else).
#     - Compute and attach content_hash, identity_key, and id via hashing.py.
#     - Attach raw_payload verbatim — do not modify or truncate it.
#     - Validate the resulting Document against the schema; raise NormalizationError if
#       any required field (url, tier, published_date, source_name) is null.
#
# This function must remain pure: same input always yields same output, no I/O, no state.
