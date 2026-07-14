"""
Contract tests for query/engine.py — the read-path orchestration.

Fully offline: a fake Chroma collection returns canned query results, a fake router returns
a fixed RoutingResult, and a fake synthesizer returns canned CLAIM text. No model loads, no
network — the same fakes-only discipline the individual phases use.

Verified:
  - route_offline / OfflineRouter build routing from the profile + query embedding, no LLM.
  - answer() with synthesizer=None runs retrieval only (answer is None, retrieval populated).
  - answer() with a synthesizer runs the full chain and grounds a claim into a citation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from query.engine import OfflineRouter, QueryResult, answer, route_offline
from retrieval.contracts import RoutingResult, UserProfile

_NOW = datetime(2026, 7, 13, tzinfo=timezone.utc)
_EPOCH = int(datetime(2026, 7, 12, tzinfo=timezone.utc).timestamp())


class FakeCollection:
    """Minimal chromadb.Collection stand-in: .query(...) returns one canned batch."""

    def __init__(self, ids, texts, metadatas):
        self._ids = ids
        self._texts = texts
        self._metadatas = metadatas
        self.last_kwargs = None

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        n = len(self._ids)
        return {
            "ids": [self._ids],
            "distances": [[0.05] * n],  # similarity = 1 - 0.05 = 0.95
            "metadatas": [self._metadatas],
            "documents": [self._texts],
            "embeddings": [[[0.1, 0.2, 0.3] for _ in range(n)]],
        }


def _fake_collection() -> FakeCollection:
    return FakeCollection(
        ids=["doc-a#0", "doc-b#0"],
        texts=[
            "Inflation in the euro area eased more than economists expected.",
            "European energy prices fell, easing the inflation outlook further.",
        ],
        metadatas=[
            {
                "source_name": "FT",
                "tier": 2,
                "published_epoch": _EPOCH,
                "url": "https://ft.com/a",
                "ordinal": 0,
            },
            {
                "source_name": "Reuters",
                "tier": 1,
                "published_epoch": _EPOCH,
                "url": "https://reuters.com/b",
                "ordinal": 0,
            },
        ],
    )


class FakeRouter:
    def __init__(self):
        self.calls = []

    def route(self, query, profile):
        self.calls.append((query, profile))
        return RoutingResult(
            intent="sector_trend",
            tickers=list(profile.tickers),
            sectors=["macro"],
            time_window_days=30,
            query_embedding=[0.1, 0.2, 0.3],
        )


class FakeSynthesizer:
    def __init__(self, reply):
        self._reply = reply
        self.prompts = []

    def synthesize(self, prompt):
        self.prompts.append(prompt)
        return self._reply


class TestRouteOffline:
    def test_builds_routing_from_profile_and_embeds_query(self):
        profile = UserProfile(tickers=["NVDA"], sectors=["semiconductors"])
        routing = route_offline("chip demand", profile, embedder=lambda t: [1.0, 2.0])
        assert routing.intent == "unknown"
        assert routing.tickers == ["NVDA"]
        assert routing.sectors == ["semiconductors"]
        assert routing.query_embedding == [1.0, 2.0]

    def test_offline_router_delegates(self):
        router = OfflineRouter(embedder=lambda t: [9.0])
        routing = router.route("q", UserProfile(tickers=["AMZN"]))
        assert routing.query_embedding == [9.0]
        assert routing.tickers == ["AMZN"]


class TestAnswerRetrievalOnly:
    def test_synthesizer_none_skips_generation(self):
        result = answer(
            "euro inflation",
            UserProfile(),
            collection=_fake_collection(),
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
        )
        assert isinstance(result, QueryResult)
        assert result.answer is None
        assert result.validated_claims is None  # no synthesis => no grounding gate output
        assert result.retrieval.chunk_count == 2
        assert result.routing.intent == "sector_trend"

    def test_search_receives_n_results(self):
        collection = _fake_collection()
        answer(
            "q",
            UserProfile(),
            collection=collection,
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
            n_results=7,
        )
        assert collection.last_kwargs["n_results"] == 7


class TestAnswerFullChain:
    def test_grounded_claim_becomes_a_citation(self):
        reply = (
            "CLAIM: Inflation in the euro area eased more than economists expected.\n"
            "SOURCE_CHUNK_ID: doc-a#0\n"
            "CONFIDENCE: high\n"
            "---"
        )
        result = answer(
            "euro inflation",
            UserProfile(),
            collection=_fake_collection(),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(reply),
            now=_NOW,
        )
        assert result.answer is not None
        assert len(result.answer.citations) == 1
        assert result.answer.citations[0].tier == 2  # FT / doc-a
        assert "euro area" in result.answer.prose
        # validated_claims exposes the grounding-gate output for observability (eval harness).
        assert result.validated_claims is not None
        assert any(c.is_grounded for c in result.validated_claims)

    def test_prompt_reaches_synthesizer(self):
        synth = FakeSynthesizer("CLAIM: x\nSOURCE_CHUNK_ID: doc-a#0\nCONFIDENCE: low\n---")
        answer(
            "q",
            UserProfile(),
            collection=_fake_collection(),
            router=FakeRouter(),
            synthesizer=synth,
            now=_NOW,
        )
        assert len(synth.prompts) == 1
        assert "doc-a#0" in synth.prompts[0]


class _WhereAwareCollection:
    """Empty when a where-clause is present, non-empty otherwise (to exercise fallback)."""

    def _empty(self):
        return {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "embeddings": [[]],
        }

    def _one(self):
        return {
            "ids": [["doc-x#0"]],
            "distances": [[0.1]],
            "metadatas": [
                [
                    {
                        "source_name": "FT",
                        "tier": 2,
                        "published_epoch": _EPOCH,
                        "url": "u",
                        "ordinal": 0,
                    }
                ]
            ],
            "documents": [["fallback pool text"]],
            "embeddings": [[[0.1, 0.2, 0.3]]],
        }

    def query(self, **kwargs):
        return self._empty() if "where" in kwargs else self._one()


class TestObservabilityFields:
    def test_trace_populates_prompt_and_raw_synthesis(self):
        reply = "CLAIM: x\nSOURCE_CHUNK_ID: doc-a#0\nCONFIDENCE: low\n---"
        result = answer(
            "q",
            UserProfile(),
            collection=_fake_collection(),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer(reply),
            now=_NOW,
            trace=True,
        )
        assert result.prompt is not None and "doc-a#0" in result.prompt
        assert result.raw_synthesis == reply

    def test_default_leaves_trace_fields_none(self):
        result = answer(
            "q",
            UserProfile(),
            collection=_fake_collection(),
            router=FakeRouter(),
            synthesizer=FakeSynthesizer("CLAIM: x\nSOURCE_CHUNK_ID: doc-a#0\nCONFIDENCE: low\n---"),
            now=_NOW,
        )
        assert result.prompt is None
        assert result.raw_synthesis is None

    def test_trace_builds_prompt_in_retrieval_only_mode(self):
        result = answer(
            "q",
            UserProfile(),
            collection=_fake_collection(),
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
            trace=True,
        )
        assert result.answer is None
        assert result.prompt is not None  # built for inspection even without Gemini
        assert result.raw_synthesis is None

    def test_filter_fallback_flag_threaded(self):
        # Router carries a time-window (where is not None); filtered query returns empty ->
        # search falls back to unfiltered -> the flag surfaces on QueryResult.
        result = answer(
            "q",
            UserProfile(),
            collection=_WhereAwareCollection(),
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
        )
        assert result.filter_fallback is True
        assert result.retrieval.chunk_count == 1  # resurrected pool

    def test_no_fallback_flag_when_filtered_nonempty(self):
        result = answer(
            "q",
            UserProfile(),
            collection=_fake_collection(),
            router=FakeRouter(),
            synthesizer=None,
            now=_NOW,
        )
        assert result.filter_fallback is False
