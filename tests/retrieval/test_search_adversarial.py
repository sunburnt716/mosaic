"""
Adversarial/edge-case tests for Phase 2 Vector Search (retrieval/search.py).

Complements test_search.py's contract tests. Drives degenerate/hostile Chroma responses and
boundary routing values at VectorSearch/build_where_clause.
"""

from __future__ import annotations

from retrieval.search import VectorSearch, build_where_clause
from tests.retrieval.conftest import FakeChromaCollection, make_query_response
from tests.retrieval.fixtures import make_routing_result


class TestWhereClauseEdgeCases:
    def test_negative_time_window_still_computes_a_cutoff(self):
        # build_where_clause trusts routing; router already rejects non-positive windows,
        # but this module shouldn't crash if handed one anyway.
        routing = make_routing_result(tickers=[], time_window_days=-10)
        where = build_where_clause(routing, now_epoch=1_000_000)
        assert where == {"published_epoch": {"$gte": 1_000_000 - (-10) * 86400}}

    def test_single_ticker_list(self):
        routing = make_routing_result(tickers=["NVDA"], time_window_days=0)
        assert build_where_clause(routing, now_epoch=0) == {"ticker": {"$in": ["NVDA"]}}

    def test_many_tickers(self):
        tickers = [f"T{i}" for i in range(50)]
        routing = make_routing_result(tickers=tickers, time_window_days=0)
        where = build_where_clause(routing, now_epoch=0)
        assert where == {"ticker": {"$in": tickers}}

    def test_now_epoch_zero_is_respected_not_treated_as_falsy_default(self):
        routing = make_routing_result(tickers=[], time_window_days=1)
        where = build_where_clause(routing, now_epoch=0)
        assert where == {"published_epoch": {"$gte": -86400}}

    def test_duplicate_tickers_passed_through_unchanged(self):
        routing = make_routing_result(tickers=["NVDA", "NVDA"], time_window_days=0)
        where = build_where_clause(routing, now_epoch=0)
        assert where == {"ticker": {"$in": ["NVDA", "NVDA"]}}


class TestVectorSearchDegenerateResponses:
    def test_missing_response_keys_yields_empty_list(self):
        collection = FakeChromaCollection({})
        assert VectorSearch(collection).search(make_routing_result()) == []

    def test_null_metadata_entry_uses_defaults(self):
        response = make_query_response(
            ids=["a#0"], distances=[0.2], metadatas=[None], documents=["text"]
        )
        collection = FakeChromaCollection(response)
        chunk = VectorSearch(collection).search(make_routing_result())[0]
        assert chunk.source_name == ""
        assert chunk.tier == 0
        assert chunk.ticker is None

    def test_distance_greater_than_one_yields_negative_similarity(self):
        response = make_query_response(
            ids=["a#0"], distances=[1.5], metadatas=[{}], documents=["text"]
        )
        collection = FakeChromaCollection(response)
        chunk = VectorSearch(collection).search(make_routing_result())[0]
        assert chunk.similarity_score == -0.5

    def test_negative_distance_yields_similarity_above_one(self):
        # Shouldn't happen with a real cosine-distance collection, but must not crash or clamp
        # silently — downstream (rerank) is responsible for any clamping policy, not this layer.
        response = make_query_response(
            ids=["a#0"], distances=[-0.2], metadatas=[{}], documents=["text"]
        )
        collection = FakeChromaCollection(response)
        chunk = VectorSearch(collection).search(make_routing_result())[0]
        assert chunk.similarity_score == 1.2

    def test_duplicate_chunk_ids_in_response_both_pass_through(self):
        response = make_query_response(
            ids=["a#0", "a#0"],
            distances=[0.1, 0.2],
            metadatas=[{}, {}],
            documents=["first", "second"],
        )
        collection = FakeChromaCollection(response)
        chunks = VectorSearch(collection).search(make_routing_result())
        assert len(chunks) == 2
        assert [c.text for c in chunks] == ["first", "second"]

    def test_empty_document_text(self):
        response = make_query_response(ids=["a#0"], distances=[0.1], metadatas=[{}], documents=[""])
        collection = FakeChromaCollection(response)
        chunk = VectorSearch(collection).search(make_routing_result())[0]
        assert chunk.text == ""

    def test_n_results_zero(self):
        collection = FakeChromaCollection(
            make_query_response(ids=[], distances=[], metadatas=[], documents=[])
        )
        VectorSearch(collection).search(make_routing_result(), n_results=0)
        assert collection.last_kwargs["n_results"] == 0

    def test_query_embedding_empty_list_still_sent(self):
        collection = FakeChromaCollection(
            make_query_response(ids=[], distances=[], metadatas=[], documents=[])
        )
        routing = make_routing_result(query_embedding=[])
        VectorSearch(collection).search(routing)
        assert collection.last_kwargs["query_embeddings"] == [[]]

    def test_ticker_metadata_missing_key_entirely(self):
        response = make_query_response(
            ids=["a#0"],
            distances=[0.1],
            metadatas=[{"source_name": "Reuters"}],  # no "ticker" key at all
            documents=["text"],
        )
        collection = FakeChromaCollection(response)
        chunk = VectorSearch(collection).search(make_routing_result())[0]
        assert chunk.ticker is None
