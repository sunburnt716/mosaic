"""
Contract tests for evals/harness.py — the behavior-first answerability eval logic.

Fully offline: a fake collection + fake router + fake synthesizer drive real read-path runs,
so the bucketing and rate math are pinned without live deps. The point is the harness's own
logic (max-not-mean similarity, behavior-first buckets, verdicts, headline rates, the
citation-suspect guard), not the quality of any real model.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from evals.harness import (
    BUCKET_CITED,
    BUCKET_NO_CANDIDATES,
    BUCKET_RETRIEVAL_ONLY,
    BUCKET_STRONG_UNCITED,
    BUCKET_SYNTH_FAILED,
    BUCKET_THIN,
    VERDICT_CORRECT_ANSWER,
    VERDICT_CORRECT_DECLINE,
    VERDICT_MISSED_ANSWER,
    VERDICT_OVER_ANSWERED,
    VERDICT_UNKNOWN,
    Question,
    evaluate,
    load_questions,
    summarize,
)
from generation.synthesizer import INSUFFICIENT_DATA_MARKER
from retrieval.contracts import RoutingResult

_NOW = datetime(2026, 7, 13, tzinfo=timezone.utc)


class FakeCollection:
    """Canned batch of `len(distances)` chunks; descending similarity = 1 - distance."""

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


class _WhereAwareCollection:
    """Empty when filtered (where present), non-empty otherwise — to exercise the fallback."""

    def query(self, **kwargs):
        if "where" in kwargs:
            return {
                "ids": [[]],
                "distances": [[]],
                "metadatas": [[]],
                "documents": [[]],
                "embeddings": [[]],
            }
        return FakeCollection([0.05]).query()


class FakeRouter:
    """Routing with a time-window (so `where` is non-None, exercising the filter path)."""

    def route(self, query, profile):
        return RoutingResult(
            intent="unknown",
            tickers=[],
            sectors=[],
            time_window_days=30,
            query_embedding=[0.1, 0.2, 0.3],
        )


def _first_handle(prompt):
    """The first source handle (S1, …) the prompt offered, or None if it offered none."""
    m = re.search(r"^CHUNK_ID: (\S+)", prompt, re.MULTILINE)
    return m.group(1) if m else None


class FakeSynthesizer:
    """Simulates a well-behaved model: cites the first source *handle* the prompt offered.

    `cite_chunk_id` (kept for call-site readability) toggles whether this fake grounds at all;
    when falsy it declines with empty output. When it grounds, it cites the handle the prompt
    actually shows — the engine translates that back to a real chunk_id, mirroring production,
    rather than a hardcoded raw chunk_id the prompt never exposes.
    """

    def __init__(self, *, cite_chunk_id=None):
        self._should_cite = bool(cite_chunk_id)

    def synthesize(self, prompt):
        if not self._should_cite:
            return ""
        handle = _first_handle(prompt)
        if handle is None:
            return ""
        return f"CLAIM: chunk text 0\nSOURCE_CHUNK_ID: {handle}\nCONFIDENCE: high\n---"


class _MarkerSynthesizer:
    """Simulates a failed Gemini *call*: returns the synthesizer's fail-closed marker verbatim,
    exactly as the real Synthesizer does after exhausting retries on a 429/503."""

    def synthesize(self, prompt):
        return INSUFFICIENT_DATA_MARKER


def _q(qid, expected, intent="news-synthesis"):
    return Question(id=qid, question=f"q {qid}", intent=intent, expected=expected)


def _run(question, collection, synthesizer):
    return evaluate(
        [question], collection=collection, router=FakeRouter(), synthesizer=synthesizer, now=_NOW
    )[0]


class TestSimilarityMetrics:
    def test_top1_is_max_top3_is_third(self):
        r = _run(_q("a", "answer"), FakeCollection([0.1, 0.2, 0.4]), None)
        assert r.top1_similarity == 0.9
        assert round(r.top3_similarity, 6) == 0.6

    def test_top3_none_when_fewer_than_three(self):
        r = _run(_q("a", "answer"), FakeCollection([0.1, 0.2]), None)
        assert r.top1_similarity == 0.9
        assert r.top3_similarity is None


class TestBehaviorFirstBuckets:
    def test_no_candidates_when_empty(self):
        # n == 0 is a filter/retrieval bug, never "thin" — and it's flagged even with synthesis.
        r = _run(_q("a", "answer"), FakeCollection([]), FakeSynthesizer(cite_chunk_id="doc-0#0"))
        assert r.n_retrieved == 0
        assert r.bucket == BUCKET_NO_CANDIDATES
        assert r.verdict == VERDICT_MISSED_ANSWER

    def test_in_scope_strong_and_cited_is_cited(self):
        r = _run(
            _q("a", "answer"),
            FakeCollection([0.05, 0.1, 0.2]),
            FakeSynthesizer(cite_chunk_id="doc-0#0"),
        )
        assert r.bucket == BUCKET_CITED
        assert r.verdict == VERDICT_CORRECT_ANSWER

    def test_in_scope_strong_but_uncited_is_the_smoking_gun(self):
        # Strong retrieval (sim 0.95 >= floor) but no citation => citation bug, not thinness.
        r = _run(_q("a", "answer"), FakeCollection([0.05]), FakeSynthesizer(cite_chunk_id=None))
        assert r.has_signal is True
        assert r.bucket == BUCKET_STRONG_UNCITED
        assert r.verdict == VERDICT_MISSED_ANSWER

    def test_in_scope_weak_and_uncited_is_thin(self):
        # sim 0.1 < floor 0.30 => genuinely thin.
        r = _run(_q("a", "answer"), FakeCollection([0.9]), FakeSynthesizer(cite_chunk_id=None))
        assert r.has_signal is False
        assert r.bucket == BUCKET_THIN
        assert r.verdict == VERDICT_MISSED_ANSWER

    def test_failed_gemini_call_is_synth_failed_not_strong_uncited(self):
        # Strong retrieval, but the Gemini *call* failed (fail-closed marker). This must NOT be
        # slandered as a citation bug (strong-uncited) — it's an infrastructure failure.
        r = _run(_q("a", "answer"), FakeCollection([0.05]), _MarkerSynthesizer())
        assert r.has_signal is True
        assert r.synthesis_failed is True
        assert r.bucket == BUCKET_SYNTH_FAILED
        assert r.verdict == VERDICT_MISSED_ANSWER  # we did miss it, just for infra reasons

    def test_out_of_scope_cited_is_over_answered(self):
        r = _run(
            _q("r", "redirect", intent="out-of-scope"),
            FakeCollection([0.05]),
            FakeSynthesizer(cite_chunk_id="doc-0#0"),
        )
        assert r.bucket == BUCKET_CITED
        assert r.verdict == VERDICT_OVER_ANSWERED

    def test_out_of_scope_declined_is_correct_decline(self):
        # Strong retrieval but correctly declined: bucket is behavioral (strong-uncited),
        # but the verdict recognizes the correct refusal — the two layers do different jobs.
        r = _run(
            _q("a1", "abstain", intent="out-of-scope"),
            FakeCollection([0.05]),
            FakeSynthesizer(cite_chunk_id=None),
        )
        assert r.bucket == BUCKET_STRONG_UNCITED
        assert r.verdict == VERDICT_CORRECT_DECLINE


class TestRetrievalOnly:
    def test_no_synthesis_is_retrieval_only(self):
        r = _run(_q("a", "answer"), FakeCollection([0.05]), None)
        assert r.synthesis_ran is False
        assert r.bucket == BUCKET_RETRIEVAL_ONLY
        assert r.verdict == VERDICT_UNKNOWN

    def test_no_candidates_observable_without_synthesis(self):
        r = _run(_q("a", "answer"), FakeCollection([]), None)
        assert r.bucket == BUCKET_NO_CANDIDATES


class TestFilterFallbackThreaded:
    def test_fallback_flag_and_resurrected_pool(self):
        r = _run(_q("a", "answer"), _WhereAwareCollection(), None)
        assert r.filter_fallback is True
        assert r.n_retrieved == 1  # resurrected via unfiltered re-query

    def test_no_fallback_when_filtered_nonempty(self):
        r = _run(_q("a", "answer"), FakeCollection([0.05]), None)
        assert r.filter_fallback is False


class TestSummary:
    def test_healthy_run_rates(self):
        # In-scope cite when the IN-SCOPE marker rides in the query; out-of-scope declines.
        class MarkerSynth:
            def synthesize(self, prompt):
                if "IN-SCOPE" in prompt:
                    handle = _first_handle(prompt)
                    return f"CLAIM: chunk text 0\nSOURCE_CHUNK_ID: {handle}\nCONFIDENCE: high\n---"
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
            synthesizer=MarkerSynth(),
            now=_NOW,
        )
        s = summarize(results)
        assert s.answerable_in_scope_rate == 1.0
        assert s.in_scope_cited == 2
        # The out-of-scope question had strong retrieval (tempted) and declined => meaningful.
        assert s.out_of_scope_tempted == 1
        assert s.meaningful_abstention_rate == 1.0
        assert s.citation_path_suspect is False

    def test_citation_suspect_flag_fires_on_in_scope_strong_uncited(self):
        # Every in-scope question retrieves strongly but nothing cites => the smoking gun.
        questions = [_q("in1", "answer"), _q("in2", "answer")]
        results = evaluate(
            questions,
            collection=FakeCollection([0.05, 0.1, 0.2]),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(cite_chunk_id=None),
            now=_NOW,
        )
        s = summarize(results)
        assert s.strong_uncited_in_scope == 2
        assert s.citation_path_suspect is True
        assert s.answerable_in_scope_rate == 0.0

    def test_synth_failure_does_not_trip_citation_suspect(self):
        # In-scope questions retrieve strongly but the Gemini call fails on every one. This is
        # an infrastructure failure, not a citation bug: it must count as synth-failed and must
        # NOT trip citation_path_suspect (the bug the whole bucket exists to prevent).
        questions = [_q("in1", "answer"), _q("in2", "answer")]
        results = evaluate(
            questions,
            collection=FakeCollection([0.05, 0.1, 0.2]),
            router=FakeRouter(),
            synthesizer=_MarkerSynthesizer(),
            now=_NOW,
        )
        s = summarize(results)
        assert s.synth_failed_count == 2
        assert s.strong_uncited_in_scope == 0
        assert s.citation_path_suspect is False
        assert s.answerable_in_scope_rate == 0.0  # still honestly missed, just not a code bug

    def test_out_of_scope_strong_uncited_does_not_trip_suspect(self):
        # An out-of-scope question that retrieves strongly and correctly declines must NOT
        # be read as a citation bug.
        questions = [_q("os1", "redirect", intent="out-of-scope")]
        results = evaluate(
            questions,
            collection=FakeCollection([0.05]),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(cite_chunk_id=None),
            now=_NOW,
        )
        s = summarize(results)
        assert s.strong_uncited_count == 1
        assert s.strong_uncited_in_scope == 0
        assert s.citation_path_suspect is False

    def test_vacuous_abstention_not_counted(self):
        # Out-of-scope question with NO retrieval (n=0): its "decline" is vacuous and must
        # not inflate the meaningful-abstention denominator.
        questions = [_q("os1", "abstain", intent="out-of-scope")]
        results = evaluate(
            questions,
            collection=FakeCollection([]),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(cite_chunk_id=None),
            now=_NOW,
        )
        s = summarize(results)
        assert s.no_candidates_count == 1
        assert s.out_of_scope_tempted == 0
        assert s.meaningful_abstention_rate is None


class TestLoadQuestions:
    def test_loads_the_real_eval_set(self):
        questions = load_questions()
        assert 20 <= len(questions) <= 40
        assert {q.intent for q in questions} == {
            "news-synthesis",
            "point-in-time-statistic",
            "ticker-specific",
            "out-of-scope",
        }
        assert {q.expected for q in questions} == {"answer", "abstain", "redirect"}
