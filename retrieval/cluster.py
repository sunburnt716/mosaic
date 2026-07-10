"""
Phase 4 — Cluster (L3 Corroboration): group the same story across outlets.

Reuses the Extraction Engine's L3 semantic-clustering logic directly — `cosine_similarity`
and `L3_SIMILARITY_THRESHOLD` from `ingestion.pipeline.dedup` (both made public there
specifically for this reuse) — rather than duplicating the math or drifting onto a different
threshold. The spec's own text approximates the threshold as "~0.92 cosine"; the actual
constant in dedup.py is 0.85, and that's what's reused here (see commit history / CLAUDE.md
for the discrepancy note) — "reuse the threshold" is read literally as importing the constant,
not re-deriving the approximation.

Clustering needs each candidate chunk's own embedding vector (Phase 4's non-goal: "no new
embedding model — reuse existing chunk vectors"), which Phase 2 now carries on
`RetrievedChunk.embedding`. A chunk with no embedding (didn't come from a collection queried
with `include=["embeddings"]`, or predates that) can't be compared and becomes its own
singleton cluster rather than raising or silently dropping.

Grouping is simple single-link clustering: a chunk joins the first existing cluster whose
primary chunk it's similar enough to; otherwise it starts a new cluster. `re-ranked` input
order means earlier chunks (higher final_score) become cluster primaries first, but the
primary within a cluster is still explicitly `tier` then `published_epoch` per the spec.

Non-goals (per spec): no cross-time-window merging (chunks are assumed temporally adjacent).
"""

from __future__ import annotations

from ingestion.pipeline.dedup import L3_SIMILARITY_THRESHOLD, cosine_similarity
from retrieval.contracts import RetrievedChunk, StoryCluster

HIGH_CORROBORATION_OUTLETS = 3  # >= this many distinct outlets => "high"


def _corroboration_label(outlet_count: int) -> str:
    if outlet_count <= 1:
        return "single"
    if outlet_count >= HIGH_CORROBORATION_OUTLETS:
        return "high"
    return "medium"


def _primary_chunk(chunks: list[RetrievedChunk]) -> RetrievedChunk:
    """Highest tier (lowest number = most trusted), then earliest published."""
    return min(chunks, key=lambda c: (c.tier, c.published_epoch))


def _build_cluster(chunks: list[RetrievedChunk]) -> StoryCluster:
    primary = _primary_chunk(chunks)
    ordered = sorted(chunks, key=lambda c: (c.tier, c.published_epoch))
    outlet_count = len({c.source_name for c in chunks})
    return StoryCluster(
        cluster_id=primary.chunk_id.split("#")[0],
        chunks=ordered,
        outlet_count=outlet_count,
        corroboration=_corroboration_label(outlet_count),
        primary_chunk=primary,
    )


class StoryClusterer:
    """Phase 4: group re-ranked RetrievedChunks into StoryClusters by semantic similarity."""

    def __init__(self, similarity_threshold: float = L3_SIMILARITY_THRESHOLD):
        self._threshold = similarity_threshold

    def cluster(self, chunks: list[RetrievedChunk]) -> list[StoryCluster]:
        groups: list[list[RetrievedChunk]] = []

        for chunk in chunks:
            joined = False
            if chunk.embedding is not None:
                for group in groups:
                    representative = group[0]
                    if representative.embedding is None:
                        continue
                    similarity = cosine_similarity(chunk.embedding, representative.embedding)
                    if similarity >= self._threshold:
                        group.append(chunk)
                        joined = True
                        break
            if not joined:
                groups.append([chunk])

        return [_build_cluster(group) for group in groups]
