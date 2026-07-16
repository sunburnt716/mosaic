"""
Phase 5 — Output Assembly: package the ranked/clustered set as a typed contract for synthesis.

No filtering happens here (spec's non-goal): every cluster the pipeline produced is handed to
synthesis, which decides what to use. Assembly only aggregates descriptive stats over the
surfaced chunks — count, outlets, time span, mean similarity, and the citation-metadata flag.

`citation_fields_present` checks `ordinal` only, not `section_label`. `ordinal` is set on every
chunk by construction (processing.chunk.build_chunk always stamps it), so `None` there can only
mean a chunk predates that fix or was never re-indexed — a real degradation worth flagging.
`section_label`, by contrast, is legitimately `None` for paragraph/fixed chunks (no section
concept for articles/tweets) as a matter of design, not a missing-data signal; requiring it
universally would make this flag False for any result set containing a non-filing chunk, which
isn't useful. An empty chunk set has nothing to cite, so the flag is False rather than
vacuously True.
"""

from __future__ import annotations

from dataclasses import dataclass

from retrieval.contracts import RetrievedChunk, StoryCluster

_SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class RetrievalOutput:
    """Phase 5 output: the typed contract handed to the Generation Pipeline."""

    clusters: list[StoryCluster]
    chunk_count: int
    outlets_represented: list[str]
    time_span_days: int
    retrieval_confidence: float  # mean similarity of surfaced chunks
    citation_fields_present: bool  # False if any surfaced chunk is missing ordinal


def _all_chunks(clusters: list[StoryCluster]) -> list[RetrievedChunk]:
    return [chunk for cluster in clusters for chunk in cluster.chunks]


def assemble_retrieval_output(clusters: list[StoryCluster]) -> RetrievalOutput:
    """Assemble a RetrievalOutput from Phase 4's clusters. Hands all clusters through untouched."""
    chunks = _all_chunks(clusters)
    chunk_count = len(chunks)

    if chunk_count == 0:
        return RetrievalOutput(
            clusters=clusters,
            chunk_count=0,
            outlets_represented=[],
            time_span_days=0,
            retrieval_confidence=0.0,
            citation_fields_present=False,
        )

    outlets_represented = sorted({chunk.source_name for chunk in chunks})
    published_epochs = [chunk.published_epoch for chunk in chunks]
    time_span_days = (max(published_epochs) - min(published_epochs)) // _SECONDS_PER_DAY
    retrieval_confidence = sum(chunk.similarity_score for chunk in chunks) / chunk_count
    citation_fields_present = all(chunk.ordinal is not None for chunk in chunks)

    return RetrievalOutput(
        clusters=clusters,
        chunk_count=chunk_count,
        outlets_represented=outlets_represented,
        time_span_days=time_span_days,
        retrieval_confidence=retrieval_confidence,
        citation_fields_present=citation_fields_present,
    )
