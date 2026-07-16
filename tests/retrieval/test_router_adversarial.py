"""
Adversarial/edge-case tests for the Phase 1 Query Router (retrieval/router.py).

Complements test_router.py's representative-query contract tests. These drive malformed,
wrong-typed, or hostile model output at QueryRouter — a real LLM can emit anything, and the
router must degrade to safe defaults rather than crash, per its "malformed output falls back
to unknown" contract.
"""

from __future__ import annotations

from retrieval.router import QueryRouter
from tests.retrieval.conftest import FakeGroqClient, make_groq_client
from tests.retrieval.fixtures import make_user_profile


class TestMalformedIntent:
    def test_intent_as_list_does_not_crash(self, fake_query_embedder):
        # A naive `intent not in VALID_INTENTS` raises TypeError on an unhashable value.
        client = make_groq_client(intent=["not", "a", "string"], tickers=[], sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("q", make_user_profile())
        assert result.intent == "unknown"

    def test_intent_as_dict_does_not_crash(self, fake_query_embedder):
        client = make_groq_client(intent={"nested": "object"}, tickers=[], sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("q", make_user_profile())
        assert result.intent == "unknown"

    def test_intent_as_int_falls_back(self, fake_query_embedder):
        client = make_groq_client(intent=42, tickers=[], sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).intent == "unknown"

    def test_intent_missing_entirely_falls_back(self, fake_query_embedder):
        client = make_groq_client(tickers=[], sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).intent == "unknown"

    def test_intent_null_falls_back(self, fake_query_embedder):
        client = make_groq_client(intent=None, tickers=[], sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).intent == "unknown"


class TestMalformedTickersAndSectors:
    def test_tickers_as_bare_string_is_rejected_not_exploded_to_chars(self, fake_query_embedder):
        # A naive `[t for t in raw["tickers"] if isinstance(t, str)]` iterates a bare string
        # char-by-char ("NVDA" -> ["N","V","D","A"]) since every char is itself a str.
        client = make_groq_client(intent="company_news", tickers="NVDA", sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("q", make_user_profile())
        assert result.tickers != ["N", "V", "D", "A"]
        assert result.tickers == []

    def test_sectors_as_bare_string_is_rejected(self, fake_query_embedder):
        client = make_groq_client(intent="company_news", tickers=[], sectors="tech")
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).sectors == []

    def test_tickers_as_dict_is_rejected(self, fake_query_embedder):
        client = make_groq_client(intent="company_news", tickers={"a": 1}, sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).tickers == []

    def test_tickers_with_mixed_types_keeps_only_strings(self, fake_query_embedder):
        client = make_groq_client(
            intent="company_news", tickers=["NVDA", 123, None, {"x": 1}, "TSMC"], sectors=[]
        )
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).tickers == ["NVDA", "TSMC"]

    def test_tickers_null_backfills_from_profile(self, fake_query_embedder):
        client = make_groq_client(intent="company_news", tickers=None, sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        profile = make_user_profile(tickers=["AAPL"])
        assert router.route("q", profile).tickers == ["AAPL"]

    def test_malformed_tickers_still_backfill_from_profile(self, fake_query_embedder):
        # Malformed ("NVDA" as bare string) collapses to [] just like an honestly empty
        # list, so profile backfill still kicks in rather than silently losing the signal.
        client = make_groq_client(intent="company_news", tickers="NVDA", sectors=[])
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        profile = make_user_profile(tickers=["AAPL"])
        assert router.route("q", profile).tickers == ["AAPL"]


class TestMalformedTimeWindow:
    def test_negative_time_window_falls_back_to_default(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=-5)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).time_window_days == 30

    def test_zero_time_window_falls_back_to_default(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=0)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).time_window_days == 30

    def test_float_time_window_falls_back_to_default(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=7.5)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).time_window_days == 30

    def test_string_time_window_falls_back_to_default(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days="30")
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).time_window_days == 30

    def test_very_large_time_window_is_accepted(self, fake_query_embedder):
        # No upper bound specified by the spec — only non-positive/wrong-typed is rejected.
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=36500)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).time_window_days == 36500


class TestHostileRawResponses:
    def test_top_level_json_array_falls_back_to_full_defaults(self, fake_query_embedder):
        client = FakeGroqClient('["not", "an", "object"]')
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("q", make_user_profile())
        assert result.intent == "unknown"
        assert result.tickers == []
        assert result.time_window_days == 30

    def test_json_scalar_falls_back_to_full_defaults(self, fake_query_embedder):
        client = FakeGroqClient("42")
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).intent == "unknown"

    def test_empty_string_response_falls_back(self, fake_query_embedder):
        client = FakeGroqClient("")
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).intent == "unknown"

    def test_prose_wrapped_json_falls_back(self, fake_query_embedder):
        # The model ignored the "no prose" instruction — must not crash trying to parse it.
        client = FakeGroqClient('Sure! Here you go: {"intent": "unknown"}')
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).intent == "unknown"

    def test_none_content_does_not_crash(self, fake_query_embedder):
        client = FakeGroqClient(None)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        assert router.route("q", make_user_profile()).intent == "unknown"


class TestQueryTextEdgeCases:
    def test_empty_query_string(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("", make_user_profile())
        assert result.query_embedding == fake_query_embedder("")

    def test_unicode_and_emoji_query(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        query = "\U0001f680 NVDA to the moon?? éèê 中文"
        result = router.route(query, make_user_profile())
        assert result.query_embedding == fake_query_embedder(query)

    def test_very_long_query_does_not_crash(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        query = "NVDA earnings " * 2000
        result = router.route(query, make_user_profile())
        assert result.query_embedding == fake_query_embedder(query)


class TestProfileEdgeCases:
    def test_both_model_and_profile_empty_yields_empty_lists(self, fake_query_embedder):
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        result = router.route("q", make_user_profile(tickers=[], sectors=[]))
        assert result.tickers == []
        assert result.sectors == []

    def test_profile_ticker_duplicates_preserved_not_deduplicated(self, fake_query_embedder):
        # Backfill is a straight copy — dedup isn't this module's job.
        client = make_groq_client(intent="unknown", tickers=[], sectors=[], time_window_days=30)
        router = QueryRouter(client=client, embedder=fake_query_embedder)
        profile = make_user_profile(tickers=["NVDA", "NVDA"])
        assert router.route("q", profile).tickers == ["NVDA", "NVDA"]
