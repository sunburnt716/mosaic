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

    def test_embeddings_requested_explicitly(self):
        # Chroma omits embeddings from query() results unless asked for; Phase 4 clustering
        # needs each chunk's own vector, so VectorSearch must always request them.
        collection = FakeChromaCollection(self._response())
        search = VectorSearch(collection)
        search.search(make_routing_result())
        assert "embeddings" in collection.last_kwargs["include"]

    def test_embedding_passed_through_when_present(self):
        response = make_query_response(
            ids=["doc-1#0"],
            distances=[0.1],
            metadatas=[{"source_name": "Reuters", "tier": 1, "published_epoch": 1, "url": "u"}],
            documents=["text"],
            embeddings=[[0.1, 0.2, 0.3]],
        )
        collection = FakeChromaCollection(response)
        chunk = VectorSearch(collection).search(make_routing_result())[0]
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_embedding_none_when_not_returned(self):
        # self._response() never sets "embeddings" — simulates a collection/response that
        # didn't include vectors; must degrade to None rather than raising.
        collection = FakeChromaCollection(self._response())
        chunk = VectorSearch(collection).search(make_routing_result())[0]
        assert chunk.embedding is None

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


class _WhereAwareCollection:
    """Returns `filtered` when a where-clause is present in the query, else `unfiltered`.

    Lets a test distinguish the filtered pass from the unfiltered re-query the fallback makes
    (the shared FakeChromaCollection returns one fixed response regardless of kwargs).
    """

    def __init__(self, filtered: dict, unfiltered: dict):
        self._filtered = filtered
        self._unfiltered = unfiltered
        self.calls: list[dict] = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self._filtered if "where" in kwargs else self._unfiltered


_EMPTY = make_query_response(ids=[], distances=[], metadatas=[], documents=[])


def _one_chunk(distance: float) -> dict:
    return make_query_response(
        ids=["doc-1#0"],
        distances=[distance],
        metadatas=[{"source_name": "FT", "tier": 2, "published_epoch": 1, "url": "u"}],
        documents=["some retrieved text"],
    )


class TestFilterFallback:
    def test_empty_filtered_falls_back_to_unfiltered(self):
        # The MSFT case: a found filter that matches zero chunks. Fallback must resurrect
        # the pool via an unfiltered re-query, not return silence.
        collection = _WhereAwareCollection(filtered=_EMPTY, unfiltered=_one_chunk(0.1))
        search = VectorSearch(collection)
        chunks = search.search(make_routing_result(tickers=["MSFT"], time_window_days=30))
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc-1#0"
        assert search.last_filter_fallback is True
        # Two queries: first with where, then without.
        assert len(collection.calls) == 2
        assert "where" in collection.calls[0]
        assert "where" not in collection.calls[1]

    def test_nonempty_filtered_keeps_filtered_no_fallback(self):
        collection = _WhereAwareCollection(filtered=_one_chunk(0.1), unfiltered=_one_chunk(0.9))
        search = VectorSearch(collection)
        chunks = search.search(make_routing_result(tickers=["NVDA"], time_window_days=30))
        assert len(chunks) == 1
        assert chunks[0].similarity_score == 0.9  # from the FILTERED response (distance 0.1)
        assert search.last_filter_fallback is False
        assert len(collection.calls) == 1

    def test_no_filter_no_fallback_even_when_empty(self):
        # where is None (no constraints): an empty result is genuinely empty, not filter
        # starvation — the fallback must not fire (there's no clause to drop).
        collection = _WhereAwareCollection(filtered=_one_chunk(0.1), unfiltered=_EMPTY)
        search = VectorSearch(collection)
        chunks = search.search(make_routing_result(tickers=[], time_window_days=0))
        assert chunks == []
        assert search.last_filter_fallback is False
        assert len(collection.calls) == 1

    def test_fallback_pool_relevance_not_inflated(self):
        # Guardrail: the fallback resurrects the pool but must not fake relevance. A weak
        # unfiltered match (distance 0.8 -> similarity 0.2) comes back low, so downstream
        # abstention still governs — the fallback restores material, it doesn't force an answer.
        collection = _WhereAwareCollection(filtered=_EMPTY, unfiltered=_one_chunk(0.8))
        search = VectorSearch(collection)
        chunks = search.search(make_routing_result(tickers=["MSFT"], time_window_days=30))
        assert search.last_filter_fallback is True
        assert abs(chunks[0].similarity_score - 0.2) < 1e-9

    def test_fallback_resets_per_search(self):
        # last_filter_fallback reflects only the most recent search (per-instance, per-query).
        collection = _WhereAwareCollection(filtered=_EMPTY, unfiltered=_one_chunk(0.1))
        search = VectorSearch(collection)
        search.search(make_routing_result(tickers=["MSFT"], time_window_days=30))
        assert search.last_filter_fallback is True
        # A second search whose filtered pass is non-empty must clear the flag.
        collection2 = _WhereAwareCollection(filtered=_one_chunk(0.1), unfiltered=_EMPTY)
        search._collection = collection2
        search.search(make_routing_result(tickers=["NVDA"], time_window_days=30))
        assert search.last_filter_fallback is False
