"""
Shared dataclasses passed between retrieval phases (Retrieval Pipeline spec).

Each phase in `retrieval/` consumes and/or produces one of these; none of them carry logic —
construction and scoring live in the phase modules (`router.py`, `search.py`, `rerank.py`,
`cluster.py`, `output.py`). Frozen, mirroring `processing.chunk.Chunk`: once a phase hands off
its result, downstream phases read it but never mutate it in place (re-rank reorders a list
rather than annotating scores back onto a RetrievedChunk).

  RoutingResult   — Phase 1 output: structured query signal + query embedding.
  RetrievedChunk  — Phase 2 output: one ANN hit with its Chroma metadata carried through.
  StoryCluster    — Phase 4 output: chunks grouped as the same story across outlets.
  UserProfile     — Phase 1 input: the ticker/sector interests used to backfill routing and
                    bias re-ranking. Not part of the spec's phase outputs, but referenced by
                    Phase 1's signature and needed somewhere; defined here since it's shared
                    input rather than any one phase's result.

`RetrievedChunk.section_label` / `.ordinal` are the metadata-dependency fields called out in
the spec: they must be passed through untouched from Chroma metadata when present (see
`processing.chunk.Chunk`, which now stamps both) so citation can locate a chunk within its
parent document. `None` on either means the source chunk predates that fix or citation is
degraded for it — retrieval still functions, it just can't cite as precisely.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserProfile:
    """A user's declared interests, used to backfill routing and bias re-ranking."""

    tickers: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingResult:
    """Phase 1 output: structured signal extracted from a freeform query."""

    intent: str  # "earnings_deep_dive" | "sector_trend" | "company_news" | "unknown"
    tickers: list[str]
    sectors: list[str]
    time_window_days: int
    query_embedding: list[float]


@dataclass(frozen=True)
class RetrievedChunk:
    """Phase 2 output: one metadata-filtered ANN hit, with citation fields passed through."""

    chunk_id: str  # "{document_id}#{ordinal}"
    text: str
    source_name: str
    tier: int  # a visible label only — never a ranking lever (spec's locked decision)
    published_epoch: int
    ticker: str | None
    similarity_score: float
    url: str
    section_label: str | None  # pass through from Chroma metadata; needed for citations
    ordinal: int | None  # pass through from Chroma metadata; needed for citations


@dataclass(frozen=True)
class StoryCluster:
    """Phase 4 output: chunks describing the same story, grouped for corroboration."""

    cluster_id: str  # derived from primary_chunk's document_id
    chunks: list[RetrievedChunk]
    outlet_count: int
    corroboration: str  # "high" | "medium" | "single"
    primary_chunk: RetrievedChunk  # highest tier, then earliest
