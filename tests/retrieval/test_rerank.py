"""
Contract tests for Phase 3 Re-rank (retrieval/rerank.py).

Pins the recency-decay anchors, profile-bias application (ticker only — sector is a
documented no-op given the locked RetrievedChunk contract has no sector field), the
final_score blend, and — critically — that tier never affects the score (spec's locked
"tier is a label, not a ranking lever" decision).
"""

from __future__ import annotations

from datetime import datetime, timezone

from retrieval.rerank import (
    PROFILE_BIAS_WEIGHT,
    RECENCY_FLOOR,
    RECENCY_FULL_SCORE,
    RECENCY_HALF_LIFE_SCORE,
    RECENCY_WEIGHT,
    RELEVANCE_WEIGHT,
    TICKER_MATCH_BIAS,
    Ranker,
    final_score,
    profile_bias,
    recency_score,
    relevance_score,
)
from tests.retrieval.fixtures import make_retrieved_chunk, make_routing_result

NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


class TestRelevanceScore:
    def test_passthrough_of_similarity_score(self):
        chunk = make_retrieved_chunk(similarity_score=0.73)
        assert relevance_score(chunk) == 0.73


class TestRecencyScore:
    def test_fresh_chunk_scores_near_one(self):
        chunk = make_retrieved_chunk(published_epoch=int(NOW.timestamp()))
        assert recency_score(chunk, NOW) == 1.0

    def test_seven_days_old_scores_point_nine_five(self):
        chunk = make_retrieved_chunk(published_epoch=int(NOW.timestamp()) - 7 * 86400)
        assert recency_score(chunk, NOW) == RECENCY_FULL_SCORE

    def test_ninety_days_old_scores_point_five(self):
        chunk = make_retrieved_chunk(published_epoch=int(NOW.timestamp()) - 90 * 86400)
        assert abs(recency_score(chunk, NOW) - RECENCY_HALF_LIFE_SCORE) < 1e-9

    def test_older_than_ninety_days_floors_out(self):
        chunk = make_retrieved_chunk(published_epoch=int(NOW.timestamp()) - 5000 * 86400)
        assert recency_score(chunk, NOW) == RECENCY_FLOOR

    def test_monotonically_decreasing_with_age(self):
        ages_days = [0, 3, 7, 30, 90, 200]
        scores = [
            recency_score(
                make_retrieved_chunk(published_epoch=int(NOW.timestamp()) - d * 86400), NOW
            )
            for d in ages_days
        ]
        assert scores == sorted(scores, reverse=True)


class TestProfileBias:
    def test_ticker_match_applies_bias(self):
        chunk = make_retrieved_chunk(ticker="NVDA")
        routing = make_routing_result(tickers=["NVDA"])
        assert profile_bias(chunk, routing) == TICKER_MATCH_BIAS

    def test_no_ticker_match_no_bias(self):
        chunk = make_retrieved_chunk(ticker="AAPL")
        routing = make_routing_result(tickers=["NVDA"])
        assert profile_bias(chunk, routing) == 0.0

    def test_chunk_with_no_ticker_no_bias(self):
        chunk = make_retrieved_chunk(ticker=None)
        routing = make_routing_result(tickers=["NVDA"])
        assert profile_bias(chunk, routing) == 0.0


class TestFinalScore:
    def test_blends_weighted_components(self):
        chunk = make_retrieved_chunk(
            similarity_score=0.8, ticker="NVDA", published_epoch=int(NOW.timestamp())
        )
        routing = make_routing_result(tickers=["NVDA"])
        expected = (
            RELEVANCE_WEIGHT * 0.8 + RECENCY_WEIGHT * 1.0 + PROFILE_BIAS_WEIGHT * TICKER_MATCH_BIAS
        )
        assert abs(final_score(chunk, routing, NOW) - expected) < 1e-9

    def test_tier_never_affects_score(self):
        base = dict(similarity_score=0.8, ticker="NVDA", published_epoch=int(NOW.timestamp()))
        routing = make_routing_result(tickers=["NVDA"])
        low_tier = make_retrieved_chunk(tier=3, **base)
        high_tier = make_retrieved_chunk(tier=0, **base)
        assert final_score(low_tier, routing, NOW) == final_score(high_tier, routing, NOW)


class TestRanker:
    def test_higher_relevance_chunk_outranks_lower_tier_chunk_regardless_of_tier(self):
        # A more-relevant Tier 3 chunk must outrank a less-relevant Tier 1 chunk
        # (spec's explicit example of the locked "tier is not a ranking lever" rule).
        routing = make_routing_result(tickers=[])
        more_relevant_tier3 = make_retrieved_chunk(
            chunk_id="a", tier=3, similarity_score=0.95, published_epoch=int(NOW.timestamp())
        )
        less_relevant_tier1 = make_retrieved_chunk(
            chunk_id="b", tier=1, similarity_score=0.4, published_epoch=int(NOW.timestamp())
        )
        ranked = Ranker().rank([less_relevant_tier1, more_relevant_tier3], routing, NOW)
        assert [c.chunk_id for c in ranked] == ["a", "b"]

    def test_rank_reorders_by_final_score_descending(self):
        routing = make_routing_result(tickers=[])
        low = make_retrieved_chunk(
            chunk_id="low", similarity_score=0.1, published_epoch=int(NOW.timestamp())
        )
        high = make_retrieved_chunk(
            chunk_id="high", similarity_score=0.9, published_epoch=int(NOW.timestamp())
        )
        ranked = Ranker().rank([low, high], routing, NOW)
        assert [c.chunk_id for c in ranked] == ["high", "low"]

    def test_empty_input_yields_empty_output(self):
        assert Ranker().rank([], make_routing_result(), NOW) == []
