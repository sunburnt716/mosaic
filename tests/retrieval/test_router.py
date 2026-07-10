"""
Contract tests for the Phase 1 Query Router (retrieval/router.py).

Drives QueryRouter with a FakeGroqClient (no network, no `groq` import) standing in for
representative Llama 3.1 8B responses, and a deterministic fake embedder standing in for
MiniLM. Covers the parsing/backfill/fallback contract, not real classification accuracy —
that needs a live-model eval set (tracked in Metrics.md, not measured here).
"""

from __future__ import annotations

import pytest

from retrieval.router import QueryRouter
from tests.retrieval.conftest import make_groq_client
from tests.retrieval.fixtures import make_user_profile


class TestRepresentativeQueries:
    """One case per representative query type, including an unclassifiable one."""

    CASES = [
        pytest.param(
            "What did NVIDIA report last earnings call?",
            dict(intent="earnings_deep_dive", tickers=["NVDA"], sectors=[], time_window_days=90),
            id="earnings_deep_dive",
        ),
        pytest.param(
            "How is the semiconductor sector trending this month?",
            dict(
                intent="sector_trend", tickers=[], sectors=["semiconductors"], time_window_days=30
            ),
            id="sector_trend",
        ),
        pytest.param(
            "Any news on Tesla today?",
            dict(intent="company_news", tickers=["TSLA"], sectors=[], time_window_days=1),
            id="company_news_ticker",
        ),
        pytest.param(
            "What's happening with TSMC and NVDA in AI chips?",
            dict(
                intent="company_news",
                tickers=["TSMC", "NVDA"],
                sectors=["semiconductors"],
                time_window_days=14,
            ),
            id="multi_ticker_with_sector",
        ),
        pytest.param(
            "Banking sector outlook for the quarter",
            dict(intent="sector_trend", tickers=[], sectors=["banking"], time_window_days=90),
            id="sector_trend_banking",
        ),
        pytest.param(
            "asdkfj random gibberish query",
            dict(intent="unknown", tickers=[], sectors=[], time_window_days=30),
            id="unknown",
        ),
    ]

    @pytest.mark.parametrize("query, model_reply", CASES)
    def test_routes_to_expected_signal(self, query, model_reply, fake_query_embedder):
        router = QueryRouter(client=make_groq_client(**model_reply), embedder=fake_query_embedder)
        result = router.route(query, make_user_profile())
        assert result.intent == model_reply["intent"]
        assert result.tickers == model_reply["tickers"]
        assert result.sectors == model_reply["sectors"]
        assert result.time_window_days == model_reply["time_window_days"]


class TestProfileBackfill:
    def test_empty_tickers_backfilled_from_profile(self, fake_query_embedder):
        client = make_groq_client(
            intent="company_news", tickers=[], sectors=[], time_window_days=30
        )
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        profile = make_user_profile(tickers=["AAPL"], sectors=["tech"])
        result = router.route("what's new", profile)
        assert result.tickers == ["AAPL"]
        assert result.sectors == ["tech"]

    def test_model_tickers_not_overridden_by_profile(self, fake_query_embedder):
        client = make_groq_client(
            intent="company_news", tickers=["NVDA"], sectors=[], time_window_days=30
        )
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        profile = make_user_profile(tickers=["AAPL"])
        result = router.route("nvda news", profile)
        assert result.tickers == ["NVDA"]


class TestFallbackBehavior:
    def test_malformed_json_falls_back_to_unknown(self, fake_query_embedder):
        from tests.retrieval.conftest import FakeGroqClient

        client = FakeGroqClient("not json at all, sorry")
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("garbled", make_user_profile())
        assert result.intent == "unknown"
        assert result.tickers == []
        assert result.sectors == []
        assert result.time_window_days == 30

    def test_invalid_intent_value_falls_back_to_unknown(self, fake_query_embedder):
        client = make_groq_client(
            intent="not_a_real_intent", tickers=[], sectors=[], time_window_days=30
        )
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("weird", make_user_profile())
        assert result.intent == "unknown"

    def test_missing_time_window_defaults_to_30(self, fake_query_embedder):
        client = make_groq_client(intent="company_news", tickers=[], sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("q", make_user_profile())
        assert result.time_window_days == 30


class TestQueryEmbedding:
    def test_embedder_called_with_raw_query(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("hello world", make_user_profile())
        assert result.query_embedding == fake_query_embedder("hello world")

    def test_embedding_uses_same_model_as_corpus(self, fake_embedder):
        # processing.utils.embedding.embed_text is the default embedder — the same function
        # the (future) corpus embedder must use, per the one-model-per-collection rule.
        from processing.utils.embedding import embed_text

        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client)
        result = router.route("hello world", make_user_profile())
        assert result.query_embedding == embed_text("hello world")


class TestPromptDiscipline:
    def test_requests_json_only_response(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        router.route("q", make_user_profile())
        kwargs = client.chat.completions.last_kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0
