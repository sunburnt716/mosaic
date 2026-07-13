"""
Contract tests for evals/harness.py — the answerability eval logic.

Fully offline: a fake collection + fake router + fake synthesizer drive real read-path
runs, so the bucketing and rate math are pinned without live deps. The point is the
harness's own logic (max-not-mean similarity, bucket assignment, headline rates), not the
quality of any real model.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evals.harness import (
    BUCKET_RETRIEVAL_ONLY,
    BUCKET_ROUTER_MISS,
    BUCKET_THIN,
    BUCKET_WORKING,
    Question,
    evaluate,
    load_questions,
    summarize,
)
from retrieval.contracts import RoutingResult

_NOW = datetime(2026, 7, 13, tzinfo=timezone.utc)


class FakeCollection:
    """Returns a canned batch of `n` chunks with descending similarity (distance ascending)."""

    def __init__(self, distances):
        self._distances = distances

    def query(self, **kwargs):
        n = len(self._distances)
        return {
            "ids": [[f"doc-{i}#0" for i in range(n)]],
            "distances": [list(self._distances)],
            "metadatas": [
                [
                    {
                        "source_name": "FT",
                        "tier": 2,
                        "published_epoch": 0,
                        "url": f"https://ft.com/{i}",
                        "ordinal": 0,
                    }
                    for i in range(n)
                ]
            ],
            "documents": [[f"chunk text {i}" for i in range(n)]],
            "embeddings": [[[0.1, 0.2, 0.3] for _ in range(n)]],
        }


class FakeRouter:
    def route(self, query, profile):
        return RoutingResult(
            intent="unknown",
            tickers=[],
            sectors=[],
            time_window_days=30,
            query_embedding=[0.1, 0.2, 0.3],
        )


class FakeSynthesizer:
    """Emits a grounded CLAIM for `answer` questions, empty text for anything else."""

    def __init__(self, *, cite_chunk_id=None):
        self._cite = cite_chunk_id

    def synthesize(self, prompt):
        if self._cite and self._cite in prompt:
            return f"CLAIM: chunk text 0\nSOURCE_CHUNK_ID: {self._cite}\nCONFIDENCE: high\n---"
        return ""  # no valid claims => honest empty state, no citation


def _q(qid, expected, intent="news-synthesis"):
    return Question(id=qid, question=f"q {qid}", intent=intent, expected=expected)


class TestSimilarityMetrics:
    def test_top1_is_max_top3_is_third(self):
        # distances 0.1/0.2/0.4 -> sims 0.9/0.8/0.6; top1=0.9, top3=0.6.
        results = evaluate(
            [_q("a", "answer")],
            collection=FakeCollection([0.1, 0.2, 0.4]),
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
        )
        r = results[0]
        assert r.top1_similarity == 0.9
        assert round(r.top3_similarity, 6) == 0.6

    def test_top3_none_when_fewer_than_three(self):
        results = evaluate(
            [_q("a", "answer")],
            collection=FakeCollection([0.1, 0.2]),
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
        )
        assert results[0].top1_similarity == 0.9
        assert results[0].top3_similarity is None


class TestRetrievalOnlyBuckets:
    def test_synthesizer_none_yields_retrieval_only_bucket(self):
        results = evaluate(
            [_q("a", "answer")],
            collection=FakeCollection([0.1]),
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
        )
        assert results[0].bucket == BUCKET_RETRIEVAL_ONLY
        assert results[0].synthesis_ran is False

    def test_has_signal_respects_floor(self):
        results = evaluate(
            [_q("a", "answer")],
            collection=FakeCollection([0.9]),  # sim = 0.1, below default floor 0.30
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
        )
        assert results[0].has_signal is False


class TestFullChainBuckets:
    def test_in_scope_with_citation_is_working(self):
        results = evaluate(
            [_q("a", "answer")],
            collection=FakeCollection([0.05, 0.1, 0.2]),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(cite_chunk_id="doc-0#0"),
            now=_NOW,
        )
        assert results[0].synthesis_citable is True
        assert results[0].validator_passed is True
        assert results[0].bucket == BUCKET_WORKING

    def test_in_scope_without_citation_is_thin(self):
        results = evaluate(
            [_q("a", "answer")],
            collection=FakeCollection([0.05]),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(cite_chunk_id=None),  # emits empty -> no claims
            now=_NOW,
        )
        assert results[0].synthesis_citable is False
        assert results[0].bucket == BUCKET_THIN

    def test_out_of_scope_answered_is_router_miss(self):
        results = evaluate(
            [_q("r", "redirect", intent="out-of-scope")],
            collection=FakeCollection([0.05]),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(cite_chunk_id="doc-0#0"),  # wrongly answers
            now=_NOW,
        )
        assert results[0].bucket == BUCKET_ROUTER_MISS

    def test_out_of_scope_declined_is_working(self):
        results = evaluate(
            [_q("a1", "abstain", intent="out-of-scope")],
            collection=FakeCollection([0.05]),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(cite_chunk_id=None),  # correctly declines
            now=_NOW,
        )
        assert results[0].bucket == BUCKET_WORKING


class TestSummary:
    def test_headline_rates_end_to_end(self):
        # A synthesizer that cites only when the query carries the IN-SCOPE marker (proxying
        # a correct answer/decline split), so both headline rates come out to 100%.
        class CiteWhenInScope:
            def synthesize(self, prompt):
                if "IN-SCOPE" in prompt:
                    return "CLAIM: chunk text 0\nSOURCE_CHUNK_ID: doc-0#0\nCONFIDENCE: high\n---"
                return ""

        questions = [
            Question(id="in1", question="IN-SCOPE one", intent="news-synthesis", expected="answer"),
            Question(id="in2", question="IN-SCOPE two", intent="news-synthesis", expected="answer"),
            Question(id="os1", question="off topic", intent="out-of-scope", expected="abstain"),
        ]
        results = evaluate(
            questions,
            collection=FakeCollection([0.05, 0.1, 0.2]),
            router=FakeRouter(),
            synthesizer=CiteWhenInScope(),
            now=_NOW,
        )
        summary = summarize(results)
        assert summary.answerable_in_scope_rate == 1.0
        assert summary.abstention_rate == 1.0
        assert summary.in_scope_total == 2
        assert summary.out_of_scope_total == 1
        assert summary.bucket_counts.get(BUCKET_WORKING) == 3


class TestLoadQuestions:
    def test_loads_the_real_eval_set(self):
        questions = load_questions()
        assert 20 <= len(questions) <= 40
        intents = {q.intent for q in questions}
        assert intents == {
            "news-synthesis",
            "point-in-time-statistic",
            "ticker-specific",
            "out-of-scope",
        }
        assert {q.expected for q in questions} == {"answer", "abstain", "redirect"}
