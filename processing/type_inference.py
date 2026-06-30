"""
Phase 0 — document type inference (algorithmic heuristics, no model).

Given a normalized Document, infer its document_type — one of "filing", "article",
"tweet", or "unknown" — from the shape of its content. This is the foundation both
ingestion (hot path) and query-time retrieval (cold path) use to pick a chunking
strategy in Phase 1, so it must be pure, deterministic, and cheap.

Hard constraint (Phase 0): inference is heuristic only. No LLM, no embeddings, no
model of any kind. MiniLM/Gemini appear in Phase 2 and are used for embeddings, never
for type detection.

How the signals combine
------------------------
Three structural signals are measured (in text_metrics): token count, distinct
filing-marker count, and paragraph count. They feed a single structural classifier
(`_structural_type`). Separately, a source may carry a human-authored advisory hint
(SourceConfig.doc_type, surfaced here as `source_hints`). Structure and advisory are
reconciled by one documented policy:

  1. A STRONG structural signal wins outright and may override the advisory:
       - filing markers + filing-scale length  -> "filing"
       - confidently tiny, single-block, marker-free text -> "tweet"
     (The advisory vocabulary cannot express "tweet" — SourceConfig.doc_type is only
     "article"/"filing" — so genuine tweets must be caught by structure even when
     their source is configured as "article".)
  2. Otherwise the advisory hint is trusted when present (config knows its sources
     better than a weak structural guess does).
  3. Otherwise fall back to the structural guess, or "unknown" if nothing fits.

Anything inference cannot place confidently becomes "unknown"; the validation gate
then records *why* rather than this module guessing. Every helper below is small and
independently testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from ingestion.core.document import Document
from processing.text_metrics import count_filing_markers, count_paragraphs, count_tokens

# ---------------------------------------------------------------------------
# Type vocabulary — the canonical set of document_type values
# ---------------------------------------------------------------------------

FILING = "filing"
ARTICLE = "article"
TWEET = "tweet"
UNKNOWN = "unknown"

# The complete, closed set of values infer_document_type can return. Validation and
# any caller normalizing an advisory hint check membership against this.
DOCUMENT_TYPES: frozenset[str] = frozenset({FILING, ARTICLE, TWEET, UNKNOWN})

# ---------------------------------------------------------------------------
# Size / structure thresholds (shared with validation, which imports them)
# ---------------------------------------------------------------------------

# Filings run large and are built from labelled sections.
FILING_MIN_TOKENS = 500
FILING_MARKER_MIN = 2  # distinct markers; one stray mention is not enough

# Tweets are tiny. TWEET_MAX_TOKENS is the structural/validation ceiling per spec;
# TWEET_STRONG_TOKENS is the tighter "this is unmistakably a tweet" ceiling at which
# structure is confident enough to override a source's "article" advisory.
TWEET_MAX_TOKENS = 280
TWEET_STRONG_TOKENS = 60

# Articles sit in a broad mid-range of prose.
ARTICLE_MIN_TOKENS = 100
ARTICLE_MAX_TOKENS = 5000


# ---------------------------------------------------------------------------
# Structural sub-signals (each independently testable)
# ---------------------------------------------------------------------------


def _looks_like_filing(token_count: int, marker_count: int) -> bool:
    """True when the text has filing-scale length AND several filing markers."""
    return marker_count >= FILING_MARKER_MIN and token_count > FILING_MIN_TOKENS


def _looks_like_tweet(token_count: int, marker_count: int, paragraph_count: int) -> bool:
    """True for short, single-block, marker-free text within the tweet ceiling."""
    return 0 < token_count < TWEET_MAX_TOKENS and marker_count == 0 and paragraph_count <= 1


def _looks_like_article(token_count: int, marker_count: int) -> bool:
    """True for mid-length prose without filing structure."""
    return (
        ARTICLE_MIN_TOKENS <= token_count <= ARTICLE_MAX_TOKENS and marker_count < FILING_MARKER_MIN
    )


def _structural_type(token_count: int, marker_count: int, paragraph_count: int) -> str:
    """Classify by structure alone, ignoring any advisory hint.

    Ordered most-specific-first: a filing signal is the strongest, then tweet, then
    the broad article range, then "unknown" for anything that fits none (e.g. a huge
    marker-free blob).
    """
    if _looks_like_filing(token_count, marker_count):
        return FILING
    if _looks_like_tweet(token_count, marker_count, paragraph_count):
        return TWEET
    if _looks_like_article(token_count, marker_count):
        return ARTICLE
    return UNKNOWN


def _advisory_type(source_name: str, source_hints: Optional[Mapping[str, str]]) -> Optional[str]:
    """Return the source's advisory document_type, or None.

    `source_hints` maps source_name -> advisory type, built by the caller from loaded
    SourceConfigs (e.g. {sc.name: sc.doc_type for sc in sources}). Unknown source
    names and out-of-vocabulary values are ignored — the hint is advisory, never
    trusted blindly.
    """
    if not source_hints:
        return None
    hint = source_hints.get(source_name)
    return hint if hint in DOCUMENT_TYPES else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def infer_document_type(
    document: Document,
    source_hints: Optional[Mapping[str, str]] = None,
) -> str:
    """Infer the document_type of a normalized Document. Pure; no I/O, no model.

    Returns one of the DOCUMENT_TYPES values. `source_hints` is the optional advisory
    map described in `_advisory_type`; omit it to rely on structure alone.

    The reconciliation policy (strong structure overrides advisory; advisory wins the
    ambiguous middle; structure is the final fallback) is documented at module level.
    """
    body = document.body or ""
    if not body.strip():
        # No content to reason about. Validation will mark this deferred.
        return UNKNOWN

    token_count = count_tokens(body)
    marker_count = count_filing_markers(body)
    paragraph_count = count_paragraphs(body)

    structural = _structural_type(token_count, marker_count, paragraph_count)

    # 1. Strong structural signals override the advisory hint.
    if structural == FILING:
        return FILING
    if structural == TWEET and token_count < TWEET_STRONG_TOKENS:
        return TWEET

    # 2. Trust the human-authored advisory hint when present.
    advisory = _advisory_type(document.source_name, source_hints)
    if advisory is not None:
        return advisory

    # 3. Fall back to the structural guess (which may itself be "unknown").
    return structural
