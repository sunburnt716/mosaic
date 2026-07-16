"""
Adversarial/edge-case tests for Phase 5 Output Assembly (retrieval/output.py).

Complements test_output.py's contract tests: degenerate/out-of-range values a real re-ranked,
clustered set (or a degenerate VectorSearch response upstream) could produce.
"""

from __future__ import annotations

from retrieval.output import assemble_retrieval_output
from tests.retrieval.fixtures import make_retrieved_chunk, make_story_cluster


class TestOutOfRangeSimilarity:
    def test_retrieval_confidence_not_clamped_to_zero_one(self):
        # search.py can hand back similarity outside [0, 1] on a malformed distance; output
        # assembly must not silently clamp/hide that, since it's a real upstream signal.
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", similarity_score=1.8),
            make_retrieved_chunk(chunk_id="b#0", similarity_score=-0.4),
        ]
        cluster = make_story_cluster(chunks=chunks)
        output = assemble_retrieval_output([cluster])
        assert abs(output.retrieval_confidence - 0.7) < 1e-9


class TestPublishedEpochEdgeCases:
    def test_negative_published_epochs_still_compute_a_valid_span(self):
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", published_epoch=-1000),
            make_retrieved_chunk(chunk_id="b#0", published_epoch=-100),
        ]
        cluster = make_story_cluster(chunks=chunks)
        output = assemble_retrieval_output([cluster])
        assert output.time_span_days == 0  # (900s span, less than one day)

    def test_all_chunks_same_epoch_zero_time_span(self):
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", published_epoch=5000),
            make_retrieved_chunk(chunk_id="b#0", published_epoch=5000),
        ]
        cluster = make_story_cluster(chunks=chunks)
        assert assemble_retrieval_output([cluster]).time_span_days == 0

    def test_single_chunk_zero_time_span(self):
        cluster = make_story_cluster(chunks=[make_retrieved_chunk(chunk_id="a#0")])
        assert assemble_retrieval_output([cluster]).time_span_days == 0


class TestDuplicateAndOverlappingClusters:
    def test_duplicate_chunk_ids_across_clusters_both_counted(self):
        # Shouldn't happen from a real pipeline (a chunk lands in exactly one cluster), but
        # assembly is a pure aggregator — it doesn't second-guess what clustering handed it.
        shared = make_retrieved_chunk(chunk_id="shared#0")
        c1 = make_story_cluster(cluster_id="c1", chunks=[shared], primary_chunk=shared)
        c2 = make_story_cluster(cluster_id="c2", chunks=[shared], primary_chunk=shared)
        output = assemble_retrieval_output([c1, c2])
        assert output.chunk_count == 2

    def test_cluster_with_empty_chunks_list_contributes_nothing(self):
        empty_cluster = make_story_cluster(
            chunks=[], primary_chunk=make_retrieved_chunk(chunk_id="ghost#0")
        )
        real_chunk = make_retrieved_chunk(chunk_id="real#0")
        real_cluster = make_story_cluster(chunks=[real_chunk])
        output = assemble_retrieval_output([empty_cluster, real_cluster])
        assert output.chunk_count == 1
        assert output.outlets_represented == [real_chunk.source_name]


class TestLargeInputs:
    def test_many_outlets_all_represented(self):
        chunks = [
            make_retrieved_chunk(chunk_id=f"{i}#0", source_name=f"Outlet{i}") for i in range(100)
        ]
        cluster = make_story_cluster(chunks=chunks)
        output = assemble_retrieval_output([cluster])
        assert len(output.outlets_represented) == 100
        assert output.chunk_count == 100
