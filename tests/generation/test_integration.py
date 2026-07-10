"""
Golden-path integration test for the Generation Pipeline: RetrievalOutput -> GeneratedAnswer,
wiring all five phases together (PromptBuilder -> Synthesizer -> ClaimParser ->
CitationValidator -> AnswerFormatter).

Fully offline — the Gemini call is a fake client (same pattern as test_synthesizer.py), and
every claim in this scenario grounds via CitationValidator's direct-lookup fast path, so the
real MiniLM embedder is never actually invoked (lazy-loaded, never triggered). Unlike
retrieval's test_integration.py, this needs no live network and is not skipped.
"""

from __future__ import annotations

from generation.claim_parser import ClaimParser
from generation.formatter import AnswerFormatter
from generation.prompt_builder import PromptBuilder
from generation.synthesizer import Synthesizer
from generation.validator import CitationValidator
from retrieval.contracts import UserProfile
from retrieval.output import RetrievalOutput
from tests.generation.test_synthesizer import _FakeGeminiClient
from tests.retrieval.fixtures import epoch, make_retrieved_chunk, make_story_cluster


class TestGoldenPath:
    def test_retrieval_output_to_generated_answer(self):
        now_epoch = epoch(2026, 7, 8)
        reuters_chunk = make_retrieved_chunk(
            chunk_id="doc-nvda-earnings#0",
            text="NVIDIA reported record quarterly revenue driven by AI chip demand.",
            source_name="Reuters",
            tier=1,
            ticker="NVDA",
            url="https://reuters.com/nvda-earnings",
            published_epoch=now_epoch,
            similarity_score=0.95,
        )
        bloomberg_chunk = make_retrieved_chunk(
            chunk_id="doc-nvda-bloomberg#0",
            text="Nvidia posted blowout earnings as data-center GPU sales surged.",
            source_name="Bloomberg",
            tier=1,
            ticker="NVDA",
            url="https://bloomberg.com/nvda-earnings",
            published_epoch=now_epoch,
            similarity_score=0.9,
        )
        cluster = make_story_cluster(
            cluster_id="doc-nvda-earnings",
            chunks=[reuters_chunk, bloomberg_chunk],
            outlet_count=2,
            corroboration="medium",
            primary_chunk=reuters_chunk,
        )
        retrieval_output = RetrievalOutput(
            clusters=[cluster],
            chunk_count=2,
            outlets_represented=["Bloomberg", "Reuters"],
            time_span_days=0,
            retrieval_confidence=0.925,
            citation_fields_present=True,
        )

        # Phase 1: Prompt Assembly.
        prompt = PromptBuilder().build(
            retrieval_output, "What's the latest on NVDA earnings?", [], UserProfile()
        )
        assert "doc-nvda-earnings#0" in prompt
        assert "doc-nvda-bloomberg#0" in prompt

        # Phase 2: Synthesis (fake Gemini client referencing the real chunk IDs from the prompt).
        gemini_reply = (
            "CLAIM: NVIDIA reported record quarterly revenue driven by AI chip demand.\n"
            "SOURCE_CHUNK_ID: doc-nvda-earnings#0\n"
            "CONFIDENCE: high\n"
            "---\n"
            "CLAIM: Data-center GPU sales surged, driving the earnings beat.\n"
            "SOURCE_CHUNK_ID: doc-nvda-bloomberg#0\n"
            "CONFIDENCE: high\n"
            "---"
        )
        raw_text = Synthesizer(client=_FakeGeminiClient([gemini_reply])).synthesize(prompt)
        assert raw_text == gemini_reply

        # Phase 3: Claim Parsing.
        claims = ClaimParser().parse(raw_text)
        assert len(claims) == 2
        assert all(claim.is_valid for claim in claims)

        # Phase 4: Citation Validation.
        chunks_by_id = {
            reuters_chunk.chunk_id: reuters_chunk,
            bloomberg_chunk.chunk_id: bloomberg_chunk,
        }
        validated_claims = CitationValidator().validate(claims, chunks_by_id)
        assert all(claim.is_grounded for claim in validated_claims)
        assert all(claim.validation_confidence == 1.0 for claim in validated_claims)

        # Phase 5: Output Formatting.
        answer = AnswerFormatter().format(validated_claims, chunks_by_id, [cluster])

        assert "NVIDIA reported record quarterly revenue" in answer.prose
        assert "Data-center GPU sales surged" in answer.prose
        assert len(answer.citations) == 2
        assert answer.confidence_warning is None  # nothing was dropped
        assert answer.corroboration_summary == {"doc-nvda-earnings": 2}

        sources = {citation.source for citation in answer.citations}
        assert sources == {"Reuters · Tier 1", "Bloomberg · Tier 1"}

        urls = {citation.url_with_fragment.split("#:~:text=")[0] for citation in answer.citations}
        assert urls == {"https://reuters.com/nvda-earnings", "https://bloomberg.com/nvda-earnings"}


class TestGoldenPathWithPartialGrounding:
    def test_hallucinated_claim_dropped_but_answer_still_produced(self):
        chunk = make_retrieved_chunk(
            chunk_id="doc-1#0",
            text="The Federal Reserve held interest rates steady this week.",
            source_name="AP",
            tier=0,
            url="https://ap.com/fed-rates",
            published_epoch=epoch(2026, 7, 8),
        )
        cluster = make_story_cluster(
            cluster_id="doc-1",
            chunks=[chunk],
            outlet_count=1,
            corroboration="single",
            primary_chunk=chunk,
        )
        retrieval_output = RetrievalOutput(
            clusters=[cluster],
            chunk_count=1,
            outlets_represented=["AP"],
            time_span_days=0,
            retrieval_confidence=0.8,
            citation_fields_present=True,
        )
        prompt = PromptBuilder().build(retrieval_output, "Fed rates?", [], UserProfile())

        # Gemini hallucinates a second claim with a chunk ID that was never retrieved, and
        # its text has zero word overlap with anything real, so the semantic fallback fails
        # it too — it should be dropped, not silently accepted.
        gemini_reply = (
            "CLAIM: The Federal Reserve held interest rates steady this week.\n"
            "SOURCE_CHUNK_ID: doc-1#0\n"
            "CONFIDENCE: high\n"
            "---\n"
            "CLAIM: The central bank also announced a surprise stock buyback program.\n"
            "SOURCE_CHUNK_ID: doc-999-nonexistent#0\n"
            "CONFIDENCE: low\n"
            "---"
        )
        raw_text = Synthesizer(client=_FakeGeminiClient([gemini_reply])).synthesize(prompt)
        claims = ClaimParser().parse(raw_text)
        chunks_by_id = {chunk.chunk_id: chunk}

        # Force the semantic fallback to fail deterministically (offline, no real MiniLM call).
        def never_matches(a, b):
            return 0.0

        validated_claims = CitationValidator(
            embedder=lambda text: [0.0], similarity_fn=never_matches
        ).validate(claims, chunks_by_id)

        assert [c.is_grounded for c in validated_claims] == [True, False]

        answer = AnswerFormatter().format(validated_claims, chunks_by_id, [cluster])
        assert "Federal Reserve held interest rates steady" in answer.prose
        assert "stock buyback" not in answer.prose
        assert len(answer.citations) == 1
        # 1 of 2 dropped = 50% > 30% threshold.
        assert answer.confidence_warning is not None


class TestGoldenPathAllHallucinated:
    def test_zero_grounded_claims_yields_honest_empty_state(self):
        chunk = make_retrieved_chunk(chunk_id="doc-1#0", text="Some real source text.")
        cluster = make_story_cluster(chunks=[chunk])
        retrieval_output = RetrievalOutput(
            clusters=[cluster],
            chunk_count=1,
            outlets_represented=["Reuters"],
            time_span_days=0,
            retrieval_confidence=0.7,
            citation_fields_present=True,
        )
        prompt = PromptBuilder().build(retrieval_output, "q", [], UserProfile())

        gemini_reply = (
            "CLAIM: A completely fabricated claim with no basis.\n"
            "SOURCE_CHUNK_ID: doc-nonexistent#0\n"
            "CONFIDENCE: low\n"
            "---"
        )
        raw_text = Synthesizer(client=_FakeGeminiClient([gemini_reply])).synthesize(prompt)
        claims = ClaimParser().parse(raw_text)
        chunks_by_id = {chunk.chunk_id: chunk}

        validated_claims = CitationValidator(
            embedder=lambda text: [0.0], similarity_fn=lambda a, b: 0.0
        ).validate(claims, chunks_by_id)
        assert all(not c.is_grounded for c in validated_claims)

        answer = AnswerFormatter().format(validated_claims, chunks_by_id, [cluster])
        from generation.formatter import EMPTY_STATE_MESSAGE

        assert answer.prose == EMPTY_STATE_MESSAGE
        assert answer.citations == []
