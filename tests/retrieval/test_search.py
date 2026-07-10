"""
Contract tests for Phase 2 Vector Search (retrieval/search.py).

Drives VectorSearch against a FakeChromaCollection (no network, no chromadb client) to pin
the where-clause construction and the metadata pass-through, especially section_label/ordinal
(the citation-metadata dependency the spec calls a blocker).
"""

from __future__ import annotations

from retrieval.search import DEFAULT_N_RESULTS, VectorSearch, build_where_clause
from tests.retrieval.conftest import FakeChromaCollection, make_query_response
from tests.retrieval.fixtures import make_routing_result


class TestBuildWhereClause:
    def test_tickers_and_time_window_combine_with_and(self):
        routing = make_routing_result(tickers=["NVDA"], time_window_days=30)
        where = build_where_clause(routing, now_epoch=1_000_000)
        assert where == {
            "$and": [
                {"ticker": {"$in": ["NVDA"]}},
                {"published_epoch": {"$gte": 1_000_000 - 30 * 86400}},
            ]
        }

    def test_tickers_only(self):
        routing = make_routing_result(tickers=["NVDA", "TSMC"], time_window_days=0)
        where = build_where_clause(routing, now_epoch=1_000_000)
        assert where == {"ticker": {"$in": ["NVDA", "TSMC"]}}

    def test_time_window_only(self):
        routing = make_routing_result(tickers=[], time_window_days=7)
        where = build_where_clause(routing, now_epoch=1_000_000)
        assert where == {"published_epoch": {"$gte": 1_000_000 - 7 * 86400}}

    def test_no_constraints_returns_none(self):
        routing = make_routing_result(tickers=[], time_window_days=0)
        assert build_where_clause(routing, now_epoch=1_000_000) is None


class TestVectorSearch:
    def _response(self, section_label=None, ordinal=0):
        return make_query_response(
            ids=["doc-1#0"],
            distances=[0.1],
            metadatas=[
                {
                    "source_name": "Reuters",
                    "tier": 1,
                    "published_epoch": 1_700_000_000,
                    "ticker": "NVDA",
                    "url": "https://example.com/a",
                    "section_label": section_label,
                    "ordinal": ordinal,
                }
            ],
            documents=["NVIDIA reported strong earnings."],
        )

    def test_similarity_score_is_one_minus_distance(self):
        collection = FakeChromaCollection(self._response())
        search = VectorSearch(collection)
        chunks = search.search(make_routing_result())
        assert chunks[0].similarity_score == 0.9

    def test_metadata_fields_pass_through(self):
        collection = FakeChromaCollection(self._response(section_label="RISK FACTORS", ordinal=3))
        search = VectorSearch(collection)
        chunks = search.search(make_routing_result())
        chunk = chunks[0]
        assert chunk.chunk_id == "doc-1#0"
        assert chunk.text == "NVIDIA reported strong earnings."
        assert chunk.source_name == "Reuters"
        assert chunk.tier == 1
        assert chunk.published_epoch == 1_700_000_000
        assert chunk.ticker == "NVDA"
        assert chunk.url == "https://example.com/a"
        assert chunk.section_label == "RISK FACTORS"
        assert chunk.ordinal == 3

    def test_missing_citation_fields_pass_through_as_none(self):
        collection = FakeChromaCollection(self._response(section_label=None, ordinal=None))
        search = VectorSearch(collection)
        chunk = search.search(make_routing_result())[0]
        assert chunk.section_label is None
        assert chunk.ordinal is None

    def test_query_embedding_and_where_clause_sent_to_collection(self):
        collection = FakeChromaCollection(self._response())
        search = VectorSearch(collection)
        routing = make_routing_result(tickers=["NVDA"], time_window_days=30)
        search.search(routing, now_epoch=1_000_000)
        kwargs = collection.last_kwargs
        assert kwargs["query_embeddings"] == [routing.query_embedding]
        assert kwargs["where"] == build_where_clause(routing, now_epoch=1_000_000)
        assert kwargs["n_results"] == DEFAULT_N_RESULTS

    def test_no_where_kwarg_when_routing_has_no_constraints(self):
        collection = FakeChromaCollection(self._response())
        search = VectorSearch(collection)
        routing = make_routing_result(tickers=[], time_window_days=0)
        search.search(routing)
        assert "where" not in collection.last_kwargs

    def test_n_results_overridable(self):
        collection = FakeChromaCollection(self._response())
        search = VectorSearch(collection)
        search.search(make_routing_result(), n_results=5)
        assert collection.last_kwargs["n_results"] == 5

    def test_empty_result_set(self):
        collection = FakeChromaCollection(
            make_query_response(ids=[], distances=[], metadatas=[], documents=[])
        )
        search = VectorSearch(collection)
        assert search.search(make_routing_result()) == []
