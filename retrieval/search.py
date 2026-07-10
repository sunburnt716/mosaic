"""
Phase 2 — Metadata-Filtered Vector Search: query Chroma with a `where` clause AND the query
embedding together.

Two mechanisms, deliberately kept separate (spec's mental model): the `where` clause is a hard
constraint deciding what's *eligible* (ticker, date window); the query embedding decides what
*ranks*, via cosine similarity computed only within that filtered subset. The filter runs first
— cheaper and more accurate than searching the whole collection and filtering after.

`collection` is any object shaped like a `chromadb.Collection` (`.query(query_embeddings=,
where=, n_results=) -> dict` with `ids`/`distances`/`metadatas`/`documents` keys, one inner list
per query). Injected rather than constructed here so tests never need a real Chroma client, and
so callers control collection lifecycle/config.

Similarity assumes Chroma's default cosine-distance space: `similarity = 1 - distance`. All
metadata fields are preserved on the way out — especially `section_label`/`ordinal`, the
citation-metadata dependency the spec calls out — via `.get(...)` defaults so a chunk missing
either (predates the Chunk schema fix, or was never re-indexed) still comes through instead of
raising.

Non-goals (per spec): no hybrid BM25/keyword search, no cross-collection querying.
"""

from __future__ import annotations

import time
from typing import Any

from retrieval.contracts import RetrievedChunk, RoutingResult

DEFAULT_N_RESULTS = 20
_SECONDS_PER_DAY = 86400


def build_where_clause(routing: RoutingResult, now_epoch: int | None = None) -> dict | None:
    """Build a Chroma `where` clause from routing's tickers + time window.

    Returns None when routing carries no constraints (an unfiltered search over the whole
    collection) rather than an empty dict, since Chroma treats `where={}` as invalid.
    """
    clauses: list[dict[str, Any]] = []
    if routing.tickers:
        clauses.append({"ticker": {"$in": routing.tickers}})
    if routing.time_window_days:
        now = now_epoch if now_epoch is not None else int(time.time())
        cutoff = now - routing.time_window_days * _SECONDS_PER_DAY
        clauses.append({"published_epoch": {"$gte": cutoff}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _to_retrieved_chunks(raw: dict[str, Any]) -> list[RetrievedChunk]:
    """Map one Chroma `query()` response (single query, batch index 0) to RetrievedChunks."""
    ids = (raw.get("ids") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]

    chunks = []
    for chunk_id, distance, metadata, text in zip(ids, distances, metadatas, documents):
        metadata = metadata or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                source_name=metadata.get("source_name", ""),
                tier=metadata.get("tier", 0),
                published_epoch=metadata.get("published_epoch", 0),
                ticker=metadata.get("ticker"),
                similarity_score=1.0 - distance,
                url=metadata.get("url", ""),
                section_label=metadata.get("section_label"),
                ordinal=metadata.get("ordinal"),
            )
        )
    return chunks


class VectorSearch:
    """Phase 2: RoutingResult -> metadata-filtered, semantically-ranked RetrievedChunks."""

    def __init__(self, collection: Any):
        self._collection = collection

    def search(
        self,
        routing: RoutingResult,
        n_results: int = DEFAULT_N_RESULTS,
        now_epoch: int | None = None,
    ) -> list[RetrievedChunk]:
        where = build_where_clause(routing, now_epoch=now_epoch)
        kwargs: dict[str, Any] = dict(
            query_embeddings=[routing.query_embedding], n_results=n_results
        )
        if where is not None:
            kwargs["where"] = where
        raw = self._collection.query(**kwargs)
        return _to_retrieved_chunks(raw)
