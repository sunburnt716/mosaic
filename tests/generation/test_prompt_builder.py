"""
Contract + adversarial tests for Phase 1 Prompt Assembly (generation/prompt_builder.py).

Covers: guardrail/format-contract presence, lens/profile rendering, cluster selection by
corroboration-strength × relevance, chunk-block rendering, and the token-budget drop policy
(lowest-ranked chunks dropped whole, never truncated mid-block).
"""

from __future__ import annotations

from generation.prompt_builder import (
    FORMAT_CONTRACT,
    GUARDRAILS,
    PromptBuilder,
    _corroboration_rank,
    _format_epoch,
    _select_top_clusters,
)
from retrieval.contracts import UserProfile
from retrieval.output import RetrievalOutput
from tests.generation.fixtures import make_lens_doc
from tests.retrieval.fixtures import epoch, make_retrieved_chunk, make_story_cluster


def _output(clusters):
    return RetrievalOutput(
        clusters=clusters,
        chunk_count=sum(len(c.chunks) for c in clusters),
        outlets_represented=[],
        time_span_days=0,
        retrieval_confidence=0.0,
        citation_fields_present=True,
    )


class TestGuardrailsAndFormatContract:
    def test_all_guardrails_present_in_prompt(self):
        prompt = PromptBuilder().build(_output([]), "q", [], UserProfile())
        for rule in GUARDRAILS:
            assert rule in prompt

    def test_format_contract_present(self):
        prompt = PromptBuilder().build(_output([]), "q", [], UserProfile())
        assert FORMAT_CONTRACT in prompt

    def test_query_included(self):
        prompt = PromptBuilder().build(_output([]), "What about NVDA?", [], UserProfile())
        assert "What about NVDA?" in prompt


class TestLensRendering:
    def test_lens_docs_included_as_framing(self):
        lens = [make_lens_doc(title="Corroboration", text="Weigh multiple sources.")]
        prompt = PromptBuilder().build(_output([]), "q", lens, UserProfile())
        assert "Corroboration" in prompt
        assert "Weigh multiple sources." in prompt
        assert "not instructions to follow prescriptively" in prompt

    def test_empty_lens_omits_framing_section(self):
        prompt = PromptBuilder().build(_output([]), "q", [], UserProfile())
        assert "INVESTING FRAMEWORK" not in prompt

    def test_multiple_lens_docs_all_included(self):
        lens = [
            make_lens_doc(title="Doc A", text="Text A"),
            make_lens_doc(title="Doc B", text="Text B"),
        ]
        prompt = PromptBuilder().build(_output([]), "q", lens, UserProfile())
        assert "Doc A" in prompt
        assert "Doc B" in prompt


class TestProfileRendering:
    def test_profile_interests_included(self):
        profile = UserProfile(tickers=["NVDA"], sectors=["semiconductors"])
        prompt = PromptBuilder().build(_output([]), "q", [], profile)
        assert "NVDA" in prompt
        assert "semiconductors" in prompt

    def test_empty_profile_omits_interests_line(self):
        prompt = PromptBuilder().build(_output([]), "q", [], UserProfile())
        assert "USER INTERESTS" not in prompt


class TestClusterSelection:
    def test_corroboration_rank_multiplies_strength_by_best_relevance(self):
        cluster = make_story_cluster(
            chunks=[
                make_retrieved_chunk(chunk_id="a#0", similarity_score=0.5),
                make_retrieved_chunk(chunk_id="a#1", similarity_score=0.9),
            ],
            corroboration="high",
        )
        assert _corroboration_rank(cluster) == 3.0 * 0.9

    def test_empty_cluster_ranks_zero(self):
        cluster = make_story_cluster(chunks=[], corroboration="high")
        assert _corroboration_rank(cluster) == 0.0

    def test_unknown_corroboration_label_ranks_zero_not_crash(self):
        cluster = make_story_cluster(
            chunks=[make_retrieved_chunk(similarity_score=0.9)], corroboration="weird_label"
        )
        assert _corroboration_rank(cluster) == 0.0

    def test_top_n_selection_orders_by_rank_descending(self):
        low = make_story_cluster(
            cluster_id="low",
            chunks=[make_retrieved_chunk(chunk_id="low#0", similarity_score=0.2)],
            corroboration="single",
        )
        high = make_story_cluster(
            cluster_id="high",
            chunks=[make_retrieved_chunk(chunk_id="high#0", similarity_score=0.9)],
            corroboration="high",
        )
        selected = _select_top_clusters([low, high], top_n=5)
        assert [c.cluster_id for c in selected] == ["high", "low"]

    def test_fewer_clusters_than_top_n_keeps_all(self):
        only = make_story_cluster(cluster_id="only")
        assert _select_top_clusters([only], top_n=5) == [only]

    def test_more_clusters_than_top_n_drops_the_rest(self):
        clusters = [
            make_story_cluster(
                cluster_id=str(i),
                chunks=[make_retrieved_chunk(chunk_id=f"{i}#0", similarity_score=i / 10)],
                corroboration="single",
            )
            for i in range(10)
        ]
        selected = _select_top_clusters(clusters, top_n=5)
        assert len(selected) == 5
        assert [c.cluster_id for c in selected] == ["9", "8", "7", "6", "5"]

    def test_chunks_from_unselected_clusters_never_appear_in_prompt(self):
        excluded_chunk = make_retrieved_chunk(chunk_id="excluded#0", similarity_score=0.01)
        included_chunk = make_retrieved_chunk(chunk_id="included#0", similarity_score=0.99)
        clusters = [
            make_story_cluster(
                cluster_id="excluded", chunks=[excluded_chunk], corroboration="single"
            ),
            make_story_cluster(
                cluster_id="included", chunks=[included_chunk], corroboration="high"
            ),
        ]
        prompt = PromptBuilder(top_clusters=1).build(_output(clusters), "q", [], UserProfile())
        assert "included#0" in prompt
        assert "excluded#0" not in prompt


class TestChunkBlockRendering:
    def test_block_contains_source_tier_chunk_id_section_text(self):
        chunk = make_retrieved_chunk(
            chunk_id="doc-1#0",
            source_name="Reuters",
            tier=1,
            published_epoch=epoch(2026, 7, 8),
            section_label="earnings_summary",
            text="NVIDIA beat estimates.",
        )
        cluster = make_story_cluster(chunks=[chunk])
        prompt = PromptBuilder().build(_output([cluster]), "q", [], UserProfile())
        assert "SOURCE: Reuters (Tier 1)" in prompt
        assert "Published: 2026-07-08" in prompt
        assert "CHUNK_ID: doc-1#0" in prompt
        assert "SECTION: earnings_summary" in prompt
        assert "TEXT: NVIDIA beat estimates." in prompt

    def test_missing_section_label_renders_as_na(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", section_label=None)
        cluster = make_story_cluster(chunks=[chunk])
        prompt = PromptBuilder().build(_output([cluster]), "q", [], UserProfile())
        assert "SECTION: n/a" in prompt

    def test_empty_chunk_text_still_renders_a_block(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", text="")
        cluster = make_story_cluster(chunks=[chunk])
        prompt = PromptBuilder().build(_output([cluster]), "q", [], UserProfile())
        assert "CHUNK_ID: a#0" in prompt


class TestFormatEpochEdgeCases:
    def test_epoch_zero(self):
        assert _format_epoch(0) == "1970-01-01"

    def test_negative_epoch_does_not_crash(self):
        assert _format_epoch(-86400) == "1969-12-31"

    def test_far_future_epoch_does_not_crash(self):
        assert _format_epoch(4_000_000_000).startswith("20")


class TestTokenBudget:
    def test_all_chunks_included_when_under_budget(self):
        chunks = [make_retrieved_chunk(chunk_id=f"{i}#0", text="short") for i in range(3)]
        cluster = make_story_cluster(chunks=chunks)
        prompt = PromptBuilder(token_budget=5000).build(_output([cluster]), "q", [], UserProfile())
        for chunk in chunks:
            assert chunk.chunk_id in prompt

    def test_over_budget_drops_lowest_ranked_chunks_first(self):
        # Two clusters, ranked high then low; a tiny budget should keep (part of) the
        # high-ranked cluster's content over the low-ranked one's.
        high_chunk = make_retrieved_chunk(chunk_id="high#0", similarity_score=0.9, text="a " * 5)
        low_chunk = make_retrieved_chunk(chunk_id="low#0", similarity_score=0.1, text="b " * 5)
        high_cluster = make_story_cluster(
            cluster_id="high", chunks=[high_chunk], corroboration="high"
        )
        low_cluster = make_story_cluster(
            cluster_id="low", chunks=[low_chunk], corroboration="single"
        )
        # Budget big enough for the header + one chunk block, not both.
        header_tokens = len(PromptBuilder().build(_output([]), "q", [], UserProfile()).split())
        one_block_tokens = 20
        builder = PromptBuilder(token_budget=header_tokens + one_block_tokens)
        prompt = builder.build(_output([high_cluster, low_cluster]), "q", [], UserProfile())
        assert "high#0" in prompt
        assert "low#0" not in prompt

    def test_zero_budget_yields_header_only_no_chunks(self):
        chunk = make_retrieved_chunk(chunk_id="a#0")
        cluster = make_story_cluster(chunks=[chunk])
        prompt = PromptBuilder(token_budget=0).build(_output([cluster]), "q", [], UserProfile())
        assert "a#0" not in prompt
        # Header content (guardrails) is still present even at zero chunk budget.
        assert GUARDRAILS[0] in prompt

    def test_dropped_chunk_never_appears_partially(self):
        # A chunk that doesn't fit contributes none of its lines, not a truncated subset.
        chunk = make_retrieved_chunk(chunk_id="dropped#0", source_name="ReallyLongSourceName")
        cluster = make_story_cluster(chunks=[chunk])
        prompt = PromptBuilder(token_budget=1).build(_output([cluster]), "q", [], UserProfile())
        assert "dropped#0" not in prompt
        assert "ReallyLongSourceName" not in prompt


class TestEmptyRetrievalOutput:
    def test_no_clusters_still_produces_valid_prompt(self):
        prompt = PromptBuilder().build(_output([]), "q", [], UserProfile())
        assert "SOURCES:" in prompt
        for rule in GUARDRAILS:
            assert rule in prompt
