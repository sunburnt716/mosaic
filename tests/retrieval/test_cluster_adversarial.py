"""
Adversarial/edge-case tests for Phase 4 Cluster (retrieval/cluster.py).

Complements test_cluster.py's contract tests: threshold boundaries, mismatched/degenerate
embeddings, and the documented founder-only comparison limitation (a chunk is only compared
against a group's first/founding member, not every member — see cluster.py's module docstring).
"""

from __future__ import annotations

import math

from ingestion.pipeline.dedup import L3_SIMILARITY_THRESHOLD
from retrieval.cluster import StoryClusterer
from tests.retrieval.fixtures import make_retrieved_chunk


def _unit_vector(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


class TestThresholdBoundary:
    def test_similarity_exactly_at_threshold_clusters(self):
        # cosine_similarity([1,0], [1,0]) == 1.0, always >= any threshold <= 1.
        a = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])
        b = make_retrieved_chunk(chunk_id="b#0", embedding=[1.0, 0.0])
        clusterer = StoryClusterer(similarity_threshold=1.0)
        assert len(clusterer.cluster([a, b])) == 1

    def test_similarity_just_below_threshold_does_not_cluster(self):
        a = make_retrieved_chunk(chunk_id="a#0", embedding=_unit_vector(0))
        b = make_retrieved_chunk(chunk_id="b#0", embedding=_unit_vector(31.9))  # cos ~0.8487
        assert len(StoryClusterer().cluster([a, b])) == 2

    def test_similarity_just_above_threshold_clusters(self):
        a = make_retrieved_chunk(chunk_id="a#0", embedding=_unit_vector(0))
        b = make_retrieved_chunk(chunk_id="b#0", embedding=_unit_vector(31.6))  # cos ~0.8516
        assert len(StoryClusterer().cluster([a, b])) == 1


class TestFounderOnlyComparison:
    def test_chunk_similar_to_non_founding_member_but_not_founder_stays_singleton(self):
        # A founds group1. B (similar to A) joins group1. C is highly similar to B but not
        # to A — true single-linkage would merge C into group1 via B, but this clusterer only
        # compares against the founder (group[0]), so C starts its own group instead.
        a = make_retrieved_chunk(chunk_id="a#0", source_name="Reuters", embedding=_unit_vector(0))
        b = make_retrieved_chunk(chunk_id="b#0", source_name="AP", embedding=_unit_vector(25.8))
        c = make_retrieved_chunk(
            chunk_id="c#0", source_name="Bloomberg", embedding=_unit_vector(45.8)
        )
        clusters = StoryClusterer().cluster([a, b, c])
        assert len(clusters) == 2
        ids_by_cluster = [{chunk.chunk_id for chunk in cl.chunks} for cl in clusters]
        assert {"a#0", "b#0"} in ids_by_cluster
        assert {"c#0"} in ids_by_cluster


class TestMismatchedOrDegenerateEmbeddings:
    def test_different_dimensionality_does_not_crash(self):
        # cosine_similarity's zip() truncates to the shorter vector rather than raising —
        # inherited from ingestion.pipeline.dedup, not re-validated here (single source of
        # truth for the reused math).
        a = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0, 0.0])
        b = make_retrieved_chunk(chunk_id="b#0", embedding=[1.0, 0.0])
        clusters = StoryClusterer().cluster([a, b])
        assert len(clusters) in (1, 2)  # must not raise; exact grouping is incidental

    def test_zero_vector_embedding_never_clusters(self):
        # cosine_similarity returns 0.0 when either norm is 0 (dedup.py's own contract).
        zero = make_retrieved_chunk(chunk_id="a#0", embedding=[0.0, 0.0])
        other = make_retrieved_chunk(chunk_id="b#0", embedding=[1.0, 0.0])
        assert len(StoryClusterer().cluster([zero, other])) == 2

    def test_opposite_vectors_never_cluster(self):
        a = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])
        b = make_retrieved_chunk(chunk_id="b#0", embedding=[-1.0, 0.0])
        assert len(StoryClusterer().cluster([a, b])) == 2

    def test_empty_embedding_list_is_treated_like_zero_vector(self):
        a = make_retrieved_chunk(chunk_id="a#0", embedding=[])
        b = make_retrieved_chunk(chunk_id="b#0", embedding=[])
        # Both present (not None) but empty; cosine_similarity divides by zero norms -> 0.0.
        assert len(StoryClusterer().cluster([a, b])) == 2


class TestPrimarySelectionTies:
    def test_identical_tier_and_published_epoch_picks_first_in_min_order(self):
        a = make_retrieved_chunk(chunk_id="a#0", tier=1, published_epoch=1000, embedding=[1.0, 0.0])
        b = make_retrieved_chunk(
            chunk_id="b#0", tier=1, published_epoch=1000, embedding=[0.99, 0.01]
        )
        cluster = StoryClusterer().cluster([a, b])[0]
        # min() is stable: ties resolve to the first candidate in iteration order.
        assert cluster.primary_chunk.chunk_id == "a#0"


class TestCorroborationBoundary:
    def test_exactly_three_outlets_is_high(self):
        chunks = [
            make_retrieved_chunk(chunk_id=f"{i}#0", source_name=name, embedding=[1.0, 0.0])
            for i, name in enumerate(["Reuters", "AP", "Bloomberg"])
        ]
        assert StoryClusterer().cluster(chunks)[0].corroboration == "high"

    def test_same_outlet_repeated_counts_as_one(self):
        chunks = [
            make_retrieved_chunk(chunk_id=f"{i}#0", source_name="Reuters", embedding=[1.0, 0.0])
            for i in range(5)
        ]
        cluster = StoryClusterer().cluster(chunks)[0]
        assert cluster.outlet_count == 1
        assert cluster.corroboration == "single"


class TestDefaultThresholdMatchesDedup:
    def test_default_matches_ingestion_constant_exactly(self):
        assert StoryClusterer()._threshold == L3_SIMILARITY_THRESHOLD
