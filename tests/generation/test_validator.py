"""
Adversarial-heavy tests for Phase 4 Citation Validation (generation/validator.py) — the
spec's own "primary user-facing quality gate... the biggest single point of failure."

Covers: direct-lookup fast path, semantic fallback (ID missing and ID hallucinated/typo'd),
the all-fail path, threshold boundaries, tie-breaking, and that a fully malformed claim never
even reaches the embedder.
"""

from __future__ import annotations

from generation.contracts import ParsedClaim
from generation.validator import (
    DIRECT_LOOKUP_CONFIDENCE,
    SEMANTIC_FALLBACK_THRESHOLD,
    UNGROUNDED_VALIDATION_CONFIDENCE,
    CitationValidator,
)
from tests.retrieval.fixtures import make_retrieved_chunk


class _SpyEmbedder:
    """Records every call so tests can assert the embedder was (or wasn't) invoked."""

    def __init__(self, vector: list[float] | None = None):
        self.calls: list[str] = []
        self._vector = vector or [1.0, 0.0]

    def __call__(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vector


def _identity_similarity(a: list[float], b: list[float]) -> float:
    """A deterministic fake similarity: 1.0 if vectors are equal, else 0.0."""
    return 1.0 if a == b else 0.0


class TestDirectLookupFastPath:
    def test_matching_chunk_id_is_grounded_with_full_confidence(self):
        chunks = {"a#0": make_retrieved_chunk(chunk_id="a#0")}
        claim = ParsedClaim(claim_text="x", source_chunk_id="a#0", confidence="high")
        result = CitationValidator().validate([claim], chunks)[0]
        assert result.is_grounded is True
        assert result.supporting_chunk_id == "a#0"
        assert result.validation_confidence == DIRECT_LOOKUP_CONFIDENCE

    def test_direct_lookup_never_calls_the_embedder(self):
        chunks = {"a#0": make_retrieved_chunk(chunk_id="a#0")}
        claim = ParsedClaim(claim_text="x", source_chunk_id="a#0", confidence="high")
        embedder = _SpyEmbedder()
        CitationValidator(embedder=embedder).validate([claim], chunks)
        assert embedder.calls == []

    def test_model_confidence_field_passed_through_unchanged(self):
        chunks = {"a#0": make_retrieved_chunk(chunk_id="a#0")}
        claim = ParsedClaim(claim_text="x", source_chunk_id="a#0", confidence="medium")
        result = CitationValidator().validate([claim], chunks)[0]
        assert result.confidence == "medium"


class TestSemanticFallbackOnMissingId:
    def test_none_source_chunk_id_falls_back_to_semantic_search(self):
        matching_chunk = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])
        chunks = {"a#0": matching_chunk}
        claim = ParsedClaim(claim_text="x", source_chunk_id=None, confidence="high")
        embedder = _SpyEmbedder(vector=[1.0, 0.0])
        result = CitationValidator(embedder=embedder, similarity_fn=_identity_similarity).validate(
            [claim], chunks
        )[0]
        assert result.is_grounded is True
        assert result.supporting_chunk_id == "a#0"
        assert embedder.calls == ["x"]


class TestSemanticFallbackOnHallucinatedId:
    def test_id_not_in_retrieved_set_falls_back_to_semantic_search(self):
        # Gemini referenced a plausible-looking but nonexistent chunk ID.
        matching_chunk = make_retrieved_chunk(chunk_id="real#0", embedding=[1.0, 0.0])
        chunks = {"real#0": matching_chunk}
        claim = ParsedClaim(claim_text="x", source_chunk_id="hallucinated#99", confidence="high")

        def near_match_similarity(a, b):
            return 0.9  # above threshold but below a direct hit's 1.0

        result = CitationValidator(
            embedder=_SpyEmbedder(vector=[1.0, 0.0]), similarity_fn=near_match_similarity
        ).validate([claim], chunks)[0]
        assert result.is_grounded is True
        assert result.supporting_chunk_id == "real#0"
        # Semantic-fallback grounding is a lower-confidence path than a direct hit.
        assert result.validation_confidence < DIRECT_LOOKUP_CONFIDENCE


class TestSemanticFallbackThresholdBoundary:
    def test_similarity_exactly_at_threshold_grounds(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])
        claim = ParsedClaim(claim_text="x", source_chunk_id=None, confidence="high")

        def fixed_similarity(a, b):
            return SEMANTIC_FALLBACK_THRESHOLD

        result = CitationValidator(
            embedder=_SpyEmbedder(), similarity_fn=fixed_similarity
        ).validate([claim], {"a#0": chunk})[0]
        assert result.is_grounded is True
        assert result.validation_confidence == SEMANTIC_FALLBACK_THRESHOLD

    def test_similarity_just_below_threshold_fails(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])
        claim = ParsedClaim(claim_text="x", source_chunk_id=None, confidence="high")

        def fixed_similarity(a, b):
            return SEMANTIC_FALLBACK_THRESHOLD - 0.0001

        result = CitationValidator(
            embedder=_SpyEmbedder(), similarity_fn=fixed_similarity
        ).validate([claim], {"a#0": chunk})[0]
        assert result.is_grounded is False
        assert result.supporting_chunk_id is None
        assert result.validation_confidence == UNGROUNDED_VALIDATION_CONFIDENCE


class TestBestMatchSelection:
    def test_picks_highest_similarity_among_multiple_chunks(self):
        chunks = {
            "low#0": make_retrieved_chunk(chunk_id="low#0", embedding=[0.0, 1.0]),
            "high#0": make_retrieved_chunk(chunk_id="high#0", embedding=[1.0, 0.0]),
        }
        claim = ParsedClaim(claim_text="x", source_chunk_id=None, confidence="high")

        def similarity(a, b):
            return 0.9 if b == [1.0, 0.0] else 0.86

        result = CitationValidator(embedder=_SpyEmbedder(), similarity_fn=similarity).validate(
            [claim], chunks
        )[0]
        assert result.supporting_chunk_id == "high#0"
        assert result.validation_confidence == 0.9

    def test_tie_breaks_to_first_inserted_chunk(self):
        chunks = {
            "first#0": make_retrieved_chunk(chunk_id="first#0", embedding=[1.0, 0.0]),
            "second#0": make_retrieved_chunk(chunk_id="second#0", embedding=[1.0, 0.0]),
        }
        claim = ParsedClaim(claim_text="x", source_chunk_id=None, confidence="high")

        def tied_similarity(a, b):
            return 0.9

        result = CitationValidator(embedder=_SpyEmbedder(), similarity_fn=tied_similarity).validate(
            [claim], chunks
        )[0]
        assert result.supporting_chunk_id == "first#0"


class TestAllFailPath:
    def test_empty_chunks_dict_fails_gracefully(self):
        claim = ParsedClaim(claim_text="x", source_chunk_id=None, confidence="high")
        result = CitationValidator(embedder=_SpyEmbedder()).validate([claim], {})[0]
        assert result.is_grounded is False
        assert result.supporting_chunk_id is None
        assert result.validation_confidence == UNGROUNDED_VALIDATION_CONFIDENCE

    def test_no_chunk_has_an_embedding_fails_gracefully(self):
        chunks = {"a#0": make_retrieved_chunk(chunk_id="a#0", embedding=None)}
        claim = ParsedClaim(claim_text="x", source_chunk_id=None, confidence="high")
        result = CitationValidator(embedder=_SpyEmbedder()).validate([claim], chunks)[0]
        assert result.is_grounded is False

    def test_wrong_id_and_weak_semantic_match_both_fail(self):
        chunk = make_retrieved_chunk(chunk_id="a#0", embedding=[0.0, 1.0])
        claim = ParsedClaim(claim_text="x", source_chunk_id="nonexistent#0", confidence="high")
        result = CitationValidator(
            embedder=_SpyEmbedder(vector=[1.0, 0.0]), similarity_fn=_identity_similarity
        ).validate([claim], {"a#0": chunk})[0]
        assert result.is_grounded is False


class TestMalformedClaimNeverReachesEmbedder:
    def test_empty_claim_text_skips_semantic_fallback_entirely(self):
        chunks = {"a#0": make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])}
        claim = ParsedClaim(claim_text="", source_chunk_id=None, confidence=None, is_valid=False)
        embedder = _SpyEmbedder()
        result = CitationValidator(embedder=embedder).validate([claim], chunks)[0]
        assert result.is_grounded is False
        assert embedder.calls == []

    def test_whitespace_only_claim_text_skips_semantic_fallback(self):
        chunks = {"a#0": make_retrieved_chunk(chunk_id="a#0", embedding=[1.0, 0.0])}
        claim = ParsedClaim(claim_text="   ", source_chunk_id=None, confidence=None)
        embedder = _SpyEmbedder()
        result = CitationValidator(embedder=embedder).validate([claim], chunks)[0]
        assert result.is_grounded is False
        assert embedder.calls == []


class TestBatchValidation:
    def test_validates_every_claim_independently(self):
        chunks = {"a#0": make_retrieved_chunk(chunk_id="a#0")}
        grounded_claim = ParsedClaim(claim_text="x", source_chunk_id="a#0", confidence="high")
        ungrounded_claim = ParsedClaim(
            claim_text="", source_chunk_id=None, confidence=None, is_valid=False
        )
        results = CitationValidator().validate([grounded_claim, ungrounded_claim], chunks)
        assert [r.is_grounded for r in results] == [True, False]

    def test_empty_claims_list_yields_empty_results(self):
        assert CitationValidator().validate([], {"a#0": make_retrieved_chunk()}) == []

    def test_order_preserved(self):
        chunks = {f"{i}#0": make_retrieved_chunk(chunk_id=f"{i}#0") for i in range(5)}
        claims = [
            ParsedClaim(claim_text=str(i), source_chunk_id=f"{i}#0", confidence="high")
            for i in range(5)
        ]
        results = CitationValidator().validate(claims, chunks)
        assert [r.supporting_chunk_id for r in results] == [f"{i}#0" for i in range(5)]


class TestDefaultsUseRealSharedInfrastructure:
    def test_default_threshold_is_the_safety_net_value(self):
        # Lowered from the spec's literal 0.85: the direct-ID (handle) path now carries the
        # common case, so this fallback is a looser safety net for a paraphrased claim whose
        # handle the model dropped/garbled (see validator.py module docstring).
        assert SEMANTIC_FALLBACK_THRESHOLD == 0.75

    def test_default_similarity_fn_is_the_reused_dedup_cosine_similarity(self):
        from ingestion.pipeline.dedup import cosine_similarity

        assert CitationValidator()._similarity_fn is cosine_similarity

    def test_default_embedder_is_the_shared_minilm_embedder(self):
        from extraction.utils.embedding import embed_text

        assert CitationValidator()._embedder is embed_text
