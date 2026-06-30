"""
Phase 0 — per-type structure validation.

After inference assigns a document_type, validation confirms the document's actual
structure matches what that type should look like. It NEVER raises and NEVER blocks:
every document keeps flowing through the pipeline. Mismatches are returned as
structured warnings so source quality can be monitored over time and later fed into
the source-validation system.

This is the safety net for inference's weakest decisions. When inference trusts a
source's advisory hint (e.g. "this source emits filings") but the body arrives with
no section headers, validation is what surfaces the contradiction — it does not try
to re-classify, only to report.

Result model
------------
validate_document returns a ValidationResult(is_valid, warnings, severity):
  - severity escalates: "info" (clean match) < "warning" (soft mismatch, still fine
    to use) < "degenerate" (structure fundamentally wrong for the type).
  - is_valid is True unless severity is "degenerate". It flags fundamentally-wrong
    documents for monitoring; it is NOT a gate — degenerate documents still flow.
  - "unknown" documents are deferred: inference could not place them, so there is no
    type-specific structure to check. We record a single deferred warning.
"""

from __future__ import annotations

from collections import namedtuple

from ingestion.core.document import Document
from processing.text_metrics import count_filing_markers, count_paragraphs, count_tokens
from processing.type_inference import (
    ARTICLE,
    ARTICLE_MAX_TOKENS,
    ARTICLE_MIN_TOKENS,
    FILING,
    FILING_MIN_TOKENS,
    TWEET,
    TWEET_MAX_TOKENS,
    UNKNOWN,
)

# Severity levels, ordered least-to-most severe.
INFO = "info"
WARNING = "warning"
DEGENERATE = "degenerate"

# Below this an "article" is too short to be meaningful prose — a degenerate case
# distinct from the merely-out-of-range warning band.
ARTICLE_DEGENERATE_TOKENS = 50

ValidationResult = namedtuple("ValidationResult", ["is_valid", "warnings", "severity"])


# ---------------------------------------------------------------------------
# Per-type checks — each returns (warnings, severity)
# ---------------------------------------------------------------------------


def _validate_filing(token_count: int, marker_count: int) -> tuple[list[str], str]:
    """Filings should carry section headers and run long."""
    if marker_count == 0:
        # The signature degenerate filing: a document typed "filing" (usually via a
        # source advisory) whose body has no section structure at all.
        return (
            [
                "filing has zero section headers (no Item/Risk Factors/MD&A markers "
                "found); body may be unstructured or extraction stripped the headings"
            ],
            DEGENERATE,
        )
    if token_count <= FILING_MIN_TOKENS:
        return (
            [
                f"filing body is short ({token_count} tokens ≤ {FILING_MIN_TOKENS}); "
                "filings are typically longer"
            ],
            WARNING,
        )
    return ([], INFO)


def _validate_article(token_count: int, paragraph_count: int) -> tuple[list[str], str]:
    """Articles should be prose in a sane length band."""
    if token_count < ARTICLE_DEGENERATE_TOKENS:
        return (
            [
                f"article body is only {token_count} tokens "
                f"(< {ARTICLE_DEGENERATE_TOKENS}); too short to be meaningful prose"
            ],
            DEGENERATE,
        )

    warnings: list[str] = []
    if token_count < ARTICLE_MIN_TOKENS or token_count > ARTICLE_MAX_TOKENS:
        warnings.append(
            f"article length {token_count} tokens is outside the expected "
            f"{ARTICLE_MIN_TOKENS}–{ARTICLE_MAX_TOKENS} token range"
        )
    if paragraph_count < 1:
        warnings.append("article has no discernible paragraphs")

    return (warnings, WARNING if warnings else INFO)


def _validate_tweet(token_count: int) -> tuple[list[str], str]:
    """Tweets should be tiny."""
    if token_count >= TWEET_MAX_TOKENS:
        return (
            [
                f"tweet is {token_count} tokens (≥ {TWEET_MAX_TOKENS}); "
                "unexpectedly long for a tweet"
            ],
            WARNING,
        )
    return ([], INFO)


def _validate_unknown() -> tuple[list[str], str]:
    """Unknown documents have no type-specific structure to check — defer."""
    return (
        ["document_type is 'unknown'; structural validation deferred (validation_deferred)"],
        WARNING,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_document(document: Document, inferred_type: str) -> ValidationResult:
    """Validate that `document`'s structure matches `inferred_type`. Never raises.

    Measures the body once and dispatches to the per-type check. An unrecognized
    inferred_type is treated like "unknown" (deferred) rather than erroring, so a
    future type can be added to inference without breaking this gate.
    """
    body = document.body or ""
    token_count = count_tokens(body)
    marker_count = count_filing_markers(body)
    paragraph_count = count_paragraphs(body)

    if inferred_type == FILING:
        warnings, severity = _validate_filing(token_count, marker_count)
    elif inferred_type == ARTICLE:
        warnings, severity = _validate_article(token_count, paragraph_count)
    elif inferred_type == TWEET:
        warnings, severity = _validate_tweet(token_count)
    elif inferred_type == UNKNOWN:
        warnings, severity = _validate_unknown()
    else:
        # Defensive: an out-of-vocabulary type cannot be structurally validated.
        warnings, severity = (
            [f"unrecognized document_type {inferred_type!r}; validation deferred"],
            WARNING,
        )

    return ValidationResult(
        is_valid=severity != DEGENERATE,
        warnings=warnings,
        severity=severity,
    )
