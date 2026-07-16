"""
Contract tests for Phase 4 Cluster (retrieval/cluster.py).

Uses explicit embedding vectors per the project convention for L3-style tests (orthogonal for
"no similarity", near-identical for "high similarity" — see CLAUDE.md's L3 testing rule),
mirroring tests/test_dedup.py's approach since this reuses the same threshold/math.
"""

from __future__ import annotations

from ingestion.pipeline.dedup import L3_SIMILARITY_THRESHOLD
from retrieval.cluster import StoryClusterer
from tests.retrieval.fixtures import epoch, make_retrieved_chunk

ORTHOGONAL_A = [1.0, 0.0]
ORTHOGONAL_B = [0.0, 1.0]
NEAR_IDENTICAL_A = [1.0, 0.0]
NEAR_IDENTICAL_B = [0.99, 0.01]


class TestSimilarChunksCluster:
    def test_near_identical_embeddings_join_one_cluster(self):
        a = make_retrieved_chunk(
            chunk_id="doc-a#0", source_name="Reuters", embedding=NEAR_IDENTICAL_A
        )
        b = make_retrieved_chunk(chunk_id="doc-b#0", source_name="AP", embedding=NEAR_IDENTICAL_B)
        clusters = StoryClusterer().cluster([a, b])
        assert len(clusters) == 1
        assert clusters[0].outlet_count == 2

    def test_orthogonal_embeddings_never_cluster(self):
        a = make_retrieved_chunk(chunk_id="doc-a#0", embedding=ORTHOGONAL_A)
        b = make_retrieved_chunk(chunk_id="doc-b#0", embedding=ORTHOGONAL_B)
        clusters = StoryClusterer().cluster([a, b])
        assert len(clusters) == 2


class TestCorroborationLabels:
    def test_single_outlet_is_singleton(self):
        chunk = make_retrieved_chunk(chunk_id="doc-a#0")
        clusters = StoryClusterer().cluster([chunk])
        assert clusters[0].corroboration == "single"
        assert clusters[0].outlet_count == 1

    def test_two_outlets_is_medium(self):
        a = make_retrieved_chunk(
            chunk_id="doc-a#0", source_name="Reuters", embedding=NEAR_IDENTICAL_A
        )
        b = make_retrieved_chunk(chunk_id="doc-b#0", source_name="AP", embedding=NEAR_IDENTICAL_B)
        clusters = StoryClusterer().cluster([a, b])
        assert clusters[0].corroboration == "medium"

    def test_three_or_more_outlets_is_high(self):
        chunks = [
            make_retrieved_chunk(chunk_id=f"doc-{i}#0", source_name=name, embedding=[1.0, 0.0])
            for i, name in enumerate(["Reuters", "AP", "Bloomberg"])
        ]
        clusters = StoryClusterer().cluster(chunks)
        assert clusters[0].corroboration == "high"
        assert clusters[0].outlet_count == 3


class TestClusterId:
    def test_derived_from_primary_chunks_document_id(self):
        chunk = make_retrieved_chunk(chunk_id="deadbeef#3")
        clusters = StoryClusterer().cluster([chunk])
        assert clusters[0].cluster_id == "deadbeef"


class TestPrimaryChunkOrdering:
    def test_lowest_tier_number_wins_as_primary(self):
        low_trust = make_retrieved_chunk(
            chunk_id="a#0", tier=3, source_name="Blog", embedding=[1.0, 0.0]
        )
        high_trust = make_retrieved_chunk(
            chunk_id="b#0", tier=0, source_name="SEC", embedding=[0.99, 0.01]
        )
        clusters = StoryClusterer().cluster([low_trust, high_trust])
        assert clusters[0].primary_chunk.chunk_id == "b#0"

    def test_tie_on_tier_breaks_by_earliest_published(self):
        later = make_retrieved_chunk(
            chunk_id="a#0",
            tier=1,
            source_name="Reuters",
            published_epoch=epoch(2026, 7, 8),
            embedding=[1.0, 0.0],
        )
        earlier = make_retrieved_chunk(
            chunk_id="b#0",
            tier=1,
            source_name="AP",
            published_epoch=epoch(2026, 7, 1),
            embedding=[0.99, 0.01],
        )
        clusters = StoryClusterer().cluster([later, earlier])
        assert clusters[0].primary_chunk.chunk_id == "b#0"

    def test_within_cluster_chunks_ordered_by_tier_then_recency(self):
        a = make_retrieved_chunk(
            chunk_id="a#0", tier=2, published_epoch=epoch(2026, 7, 5), embedding=[1.0, 0.0]
        )
        b = make_retrieved_chunk(
            chunk_id="b#0", tier=0, published_epoch=epoch(2026, 7, 8), embedding=[0.99, 0.01]
        )
        c = make_retrieved_chunk(
            chunk_id="c#0", tier=2, published_epoch=epoch(2026, 7, 1), embedding=[0.98, 0.02]
        )
        clusters = StoryClusterer().cluster([a, b, c])
        assert [c.chunk_id for c in clusters[0].chunks] == ["b#0", "c#0", "a#0"]


class TestMissingEmbeddings:
    def test_chunk_without_embedding_becomes_own_singleton(self):
        with_embedding = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])
        without_embedding = make_retrieved_chunk(chunk_id="b#0", embedding=None)
        clusters = StoryClusterer().cluster([with_embedding, without_embedding])
        assert len(clusters) == 2

    def test_all_missing_embeddings_all_singletons(self):
        chunks = [make_retrieved_chunk(chunk_id=f"{i}#0", embedding=None) for i in range(3)]
        clusters = StoryClusterer().cluster(chunks)
        assert len(clusters) == 3
        assert all(c.corroboration == "single" for c in clusters)


class TestThresholdReuse:
    def test_default_threshold_is_the_ingestion_l3_constant(self):
        assert StoryClusterer()._threshold == L3_SIMILARITY_THRESHOLD

    def test_custom_threshold_overridable(self):
        a = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])
        b = make_retrieved_chunk(chunk_id="b#0", embedding=[0.6, 0.8])  # cosine similarity 0.6
        assert len(StoryClusterer().cluster([a, b])) == 2  # below default 0.85 threshold
        loose = StoryClusterer(similarity_threshold=0.5)
        assert len(loose.cluster([a, b])) == 1  # above the looser 0.5 threshold


class TestEmptyInput:
    def test_no_chunks_yields_no_clusters(self):
        assert StoryClusterer().cluster([]) == []
