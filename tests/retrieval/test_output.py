"""
Contract tests for Phase 5 Output Assembly (retrieval/output.py).

Pins the aggregate stats (chunk_count, outlets, time span, mean similarity) and the
citation_fields_present flag's ordinal-only check (section_label is legitimately None for
non-filing chunks by design, so it isn't part of the degradation signal — see module docstring).
"""

from __future__ import annotations

from retrieval.output import assemble_retrieval_output
from tests.retrieval.fixtures import epoch, make_retrieved_chunk, make_story_cluster


class TestAggregateStats:
    def test_chunk_count_sums_across_clusters(self):
        c1 = make_story_cluster(
            chunks=[make_retrieved_chunk(chunk_id="a#0"), make_retrieved_chunk(chunk_id="a#1")]
        )
        c2 = make_story_cluster(chunks=[make_retrieved_chunk(chunk_id="b#0")])
        output = assemble_retrieval_output([c1, c2])
        assert output.chunk_count == 3

    def test_outlets_represented_deduplicated_and_sorted(self):
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", source_name="Reuters"),
            make_retrieved_chunk(chunk_id="b#0", source_name="AP"),
            make_retrieved_chunk(chunk_id="c#0", source_name="Reuters"),
        ]
        cluster = make_story_cluster(chunks=chunks)
        output = assemble_retrieval_output([cluster])
        assert output.outlets_represented == ["AP", "Reuters"]

    def test_time_span_days_from_min_to_max_published(self):
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", published_epoch=epoch(2026, 7, 1)),
            make_retrieved_chunk(chunk_id="b#0", published_epoch=epoch(2026, 7, 8)),
        ]
        cluster = make_story_cluster(chunks=chunks)
        output = assemble_retrieval_output([cluster])
        assert output.time_span_days == 7

    def test_retrieval_confidence_is_mean_similarity(self):
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", similarity_score=0.8),
            make_retrieved_chunk(chunk_id="b#0", similarity_score=0.4),
        ]
        cluster = make_story_cluster(chunks=chunks)
        output = assemble_retrieval_output([cluster])
        assert abs(output.retrieval_confidence - 0.6) < 1e-9

    def test_clusters_passed_through_unfiltered(self):
        c1 = make_story_cluster(chunks=[make_retrieved_chunk(chunk_id="a#0")])
        c2 = make_story_cluster(chunks=[make_retrieved_chunk(chunk_id="b#0")])
        output = assemble_retrieval_output([c1, c2])
        assert output.clusters == [c1, c2]


class TestCitationFieldsPresent:
    def test_true_when_every_chunk_has_ordinal(self):
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", ordinal=0, section_label=None),
            make_retrieved_chunk(chunk_id="a#1", ordinal=1, section_label="RISK FACTORS"),
        ]
        cluster = make_story_cluster(chunks=chunks)
        assert assemble_retrieval_output([cluster]).citation_fields_present is True

    def test_false_when_any_chunk_missing_ordinal(self):
        chunks = [
            make_retrieved_chunk(chunk_id="a#0", ordinal=0),
            make_retrieved_chunk(chunk_id="a#1", ordinal=None),
        ]
        cluster = make_story_cluster(chunks=chunks)
        assert assemble_retrieval_output([cluster]).citation_fields_present is False

    def test_missing_section_label_alone_does_not_degrade_flag(self):
        # section_label is legitimately None for paragraph/fixed chunks — not a missing-data
        # signal on its own.
        chunk = make_retrieved_chunk(chunk_id="a#0", ordinal=0, section_label=None)
        cluster = make_story_cluster(chunks=[chunk])
        assert assemble_retrieval_output([cluster]).citation_fields_present is True


class TestEmptyInput:
    def test_no_clusters_yields_zeroed_output(self):
        output = assemble_retrieval_output([])
        assert output.chunk_count == 0
        assert output.outlets_represented == []
        assert output.time_span_days == 0
        assert output.retrieval_confidence == 0.0
        assert output.citation_fields_present is False
        assert output.clusters == []
