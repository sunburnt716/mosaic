# Defines the Adapter abstract base class — the interface every format adapter must implement.
#
# An adapter's sole job is to fetch raw data from one source and yield minimally-parsed items.
# It does NOT normalize, hash, dedup, or store — those are downstream pipeline concerns.
# Keeping adapters narrow ensures that adding a new source format = implementing one method.
#
# Interface:
#   fetch(config: SourceConfig) -> Iterable[dict]
#     Given a fully-loaded SourceConfig, fetch from the source and yield raw item dicts.
#     Each dict is the adapter's best-effort parse of a single article, filing, or entry.
#     The dict shape is adapter-specific; normalizer.py is responsible for mapping it to Document.
#
# Contract adapters must honour:
#   - Yield one dict per logical document (one article, one filing, one post).
#   - Always include at minimum: a raw URL, a raw publication timestamp, and the raw text or body.
#   - Never modify the source payload — pass it through as-is so raw_payload can be preserved.
#   - Raise a typed FetchError (not a bare exception) on network or parse failure so the engine
#     can isolate the failing source without aborting the entire run.
#   - Be stateless: all per-source config comes from SourceConfig, not instance variables set
#     outside __init__. This lets the engine safely re-instantiate adapters across runs.
