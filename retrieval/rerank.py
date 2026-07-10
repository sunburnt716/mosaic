"""
Phase 3 — Re-rank: reorder the raw ANN results on non-semantic signals.

Static, hand-tuned weights (spec's locked decision) — transparent and testable, not learned.
All constants live here, in one place, so tuning later means touching one module.

`final_score = RELEVANCE_WEIGHT * relevance + RECENCY_WEIGHT * recency
             + PROFILE_BIAS_WEIGHT * profile_bias`

**Tier is never scored.** It is carried through on `RetrievedChunk.tier` as a visible label
only (spec's locked decision) — a highly-relevant low-tier chunk must be able to outrank a
less-relevant high-tier one; trust is surfaced to the user at synthesis time, never used here
to silently bury a result.

`profile_bias`'s "+0.05 if sector matches" (spec text) is a documented no-op: `RetrievedChunk`
(the Phase 2 output contract, locked) carries no sector field to compare against — only
`ticker`. `SECTOR_MATCH_BIAS` is defined for when that changes, but nothing applies it yet.
Only the ticker-match bias is live.

Non-goals (per spec): no learning-to-rank, no learned weights, no tier-based boosting.
"""

from __future__ import annotations

from datetime import datetime

from retrieval.contracts import RetrievedChunk, RoutingResult

RELEVANCE_WEIGHT = 0.5
RECENCY_WEIGHT = 0.3
PROFILE_BIAS_WEIGHT = 0.2

# Recency decay anchors: ~1.0 fresh, 0.95 by RECENCY_FULL_DAYS, ~0.5 by RECENCY_HALF_LIFE_DAYS,
# floored at RECENCY_FLOOR for anything older (piecewise-linear between anchors).
RECENCY_FULL_DAYS = 7
RECENCY_FULL_SCORE = 0.95
RECENCY_HALF_LIFE_DAYS = 90
RECENCY_HALF_LIFE_SCORE = 0.5
RECENCY_FLOOR = 0.1

TICKER_MATCH_BIAS = 0.1
SECTOR_MATCH_BIAS = 0.05  # not yet applied — see module docstring


def relevance_score(chunk: RetrievedChunk) -> float:
    """The original cosine similarity from Phase 2, unchanged."""
    return chunk.similarity_score


def recency_score(chunk: RetrievedChunk, now: datetime) -> float:
    """Piecewise-linear recency decay: ~1.0 fresh, decaying to a floor by 90+ days old."""
    age_days = max(0.0, (now.timestamp() - chunk.published_epoch) / 86400)

    if age_days <= RECENCY_FULL_DAYS:
        span_score = 1.0 - RECENCY_FULL_SCORE
        return 1.0 - span_score * (age_days / RECENCY_FULL_DAYS)

    if age_days <= RECENCY_HALF_LIFE_DAYS:
        span_days = RECENCY_HALF_LIFE_DAYS - RECENCY_FULL_DAYS
        span_score = RECENCY_FULL_SCORE - RECENCY_HALF_LIFE_SCORE
        return RECENCY_FULL_SCORE - span_score * ((age_days - RECENCY_FULL_DAYS) / span_days)

    span_days = RECENCY_HALF_LIFE_DAYS - RECENCY_FULL_DAYS
    slope = (RECENCY_FULL_SCORE - RECENCY_HALF_LIFE_SCORE) / span_days
    extra_days = age_days - RECENCY_HALF_LIFE_DAYS
    return max(RECENCY_FLOOR, RECENCY_HALF_LIFE_SCORE - slope * extra_days)


def profile_bias(chunk: RetrievedChunk, routing: RoutingResult) -> float:
    """+TICKER_MATCH_BIAS when the chunk's ticker is one routing resolved to (query or profile)."""
    bias = 0.0
    if chunk.ticker and chunk.ticker in routing.tickers:
        bias += TICKER_MATCH_BIAS
    return bias


def final_score(chunk: RetrievedChunk, routing: RoutingResult, now: datetime) -> float:
    """Blend relevance, recency, and profile bias. Tier never contributes (see module docstring)."""
    return (
        RELEVANCE_WEIGHT * relevance_score(chunk)
        + RECENCY_WEIGHT * recency_score(chunk, now)
        + PROFILE_BIAS_WEIGHT * profile_bias(chunk, routing)
    )


class Ranker:
    """Phase 3: reorder RetrievedChunks by final_score, highest first."""

    def rank(
        self, chunks: list[RetrievedChunk], routing: RoutingResult, now: datetime
    ) -> list[RetrievedChunk]:
        return sorted(chunks, key=lambda c: final_score(c, routing, now), reverse=True)
