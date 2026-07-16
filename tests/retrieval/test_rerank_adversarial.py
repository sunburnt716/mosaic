"""
Adversarial/edge-case tests for Phase 3 Re-rank (retrieval/rerank.py).

Complements test_rerank.py's contract tests: boundary/extreme values a real ANN result or a
degenerate VectorSearch response (see test_search_adversarial.py) could hand this phase.
"""

from __future__ import annotations

from datetime import datetime, timezone

from retrieval.rerank import Ranker, final_score, profile_bias, recency_score
from tests.retrieval.fixtures import make_retrieved_chunk, make_routing_result

NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


class TestRecencyScoreExtremes:
    def test_future_published_epoch_clamps_to_fresh_not_above_one(self):
        chunk = make_retrieved_chunk(published_epoch=int(NOW.timestamp()) + 10 * 86400)
        assert recency_score(chunk, NOW) == 1.0

    def test_epoch_zero_still_computes_a_score(self):
        chunk = make_retrieved_chunk(published_epoch=0)
        assert 0.0 <= recency_score(chunk, NOW) <= 1.0

    def test_negative_epoch_does_not_crash(self):
        chunk = make_retrieved_chunk(published_epoch=-1_000_000_000)
        assert recency_score(chunk, NOW) == recency_score(
            make_retrieved_chunk(published_epoch=-2_000_000_000), NOW
        )  # both floored, so equal


class TestProfileBiasEdgeCases:
    def test_ticker_matches_are_case_sensitive(self):
        # Deliberate: routing.tickers come from an uppercase-instructed model/profile, so a
        # lowercase chunk ticker is treated as a genuine non-match, not normalized.
        chunk = make_retrieved_chunk(ticker="nvda")
        routing = make_routing_result(tickers=["NVDA"])
        assert profile_bias(chunk, routing) == 0.0

    def test_empty_ticker_string_never_matches(self):
        chunk = make_retrieved_chunk(ticker="")
        routing = make_routing_result(tickers=[""])
        # "" is falsy, so `chunk.ticker and ...` short-circuits — an empty-string ticker
        # never contributes bias even if routing "matches" it.
        assert profile_bias(chunk, routing) == 0.0

    def test_duplicate_routing_tickers_do_not_double_bias(self):
        chunk = make_retrieved_chunk(ticker="NVDA")
        routing = make_routing_result(tickers=["NVDA", "NVDA", "NVDA"])
        assert profile_bias(chunk, routing) == 0.1

    def test_empty_routing_tickers_no_bias(self):
        chunk = make_retrieved_chunk(ticker="NVDA")
        routing = make_routing_result(tickers=[])
        assert profile_bias(chunk, routing) == 0.0


class TestFinalScoreExtremeInputs:
    def test_similarity_score_out_of_zero_one_range_still_blends(self):
        # search.py can hand back similarity outside [0, 1] on a malformed distance
        # (see test_search_adversarial.py); rerank must not crash on it.
        chunk = make_retrieved_chunk(
            similarity_score=1.5, ticker=None, published_epoch=int(NOW.timestamp())
        )
        routing = make_routing_result(tickers=[])
        score = final_score(chunk, routing, NOW)
        assert score > 0.5 * 1.0  # relevance term alone already exceeds a normal max

    def test_negative_similarity_score_still_blends(self):
        chunk = make_retrieved_chunk(
            similarity_score=-0.5, ticker=None, published_epoch=int(NOW.timestamp())
        )
        routing = make_routing_result(tickers=[])
        assert final_score(chunk, routing, NOW) < 0.5


class TestRankerEdgeCases:
    def test_all_equal_scores_preserve_original_order(self):
        routing = make_routing_result(tickers=[])
        a = make_retrieved_chunk(
            chunk_id="a", similarity_score=0.5, published_epoch=int(NOW.timestamp())
        )
        b = make_retrieved_chunk(
            chunk_id="b", similarity_score=0.5, published_epoch=int(NOW.timestamp())
        )
        c = make_retrieved_chunk(
            chunk_id="c", similarity_score=0.5, published_epoch=int(NOW.timestamp())
        )
        ranked = Ranker().rank([a, b, c], routing, NOW)
        assert [x.chunk_id for x in ranked] == ["a", "b", "c"]

    def test_single_chunk_input(self):
        chunk = make_retrieved_chunk(chunk_id="only")
        ranked = Ranker().rank([chunk], make_routing_result(), NOW)
        assert [c.chunk_id for c in ranked] == ["only"]

    def test_does_not_mutate_input_list(self):
        routing = make_routing_result(tickers=[])
        low = make_retrieved_chunk(
            chunk_id="low", similarity_score=0.1, published_epoch=int(NOW.timestamp())
        )
        high = make_retrieved_chunk(
            chunk_id="high", similarity_score=0.9, published_epoch=int(NOW.timestamp())
        )
        original = [low, high]
        Ranker().rank(original, routing, NOW)
        assert [c.chunk_id for c in original] == ["low", "high"]

    def test_duplicate_chunk_ids_both_survive_ranking(self):
        routing = make_routing_result(tickers=[])
        a = make_retrieved_chunk(
            chunk_id="dup", similarity_score=0.9, published_epoch=int(NOW.timestamp())
        )
        b = make_retrieved_chunk(
            chunk_id="dup", similarity_score=0.1, published_epoch=int(NOW.timestamp())
        )
        ranked = Ranker().rank([b, a], routing, NOW)
        assert len(ranked) == 2
        assert ranked[0].similarity_score == 0.9
