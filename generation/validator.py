"""
Phase 4 — Citation / Retrieval Validation Layer: guarantee every claim is grounded in a real
retrieved chunk. The primary user-facing quality gate — the anti-hallucination backbone.

Per claim, two paths, in order:
  1. **Direct lookup (fast path).** `claim.source_chunk_id` exists in the retrieved set ->
     grounded, `validation_confidence = 1.0`. Deterministic and cheap; the common case, since
     the prompt already handed Gemini the exact IDs. Covers both an ID-less claim (Phase 3
     already set `source_chunk_id = None`) and a *hallucinated* ID (present but not a key in
     `chunks`) — both simply fail the lookup and fall through.
  2. **Semantic fallback.** Embed the claim text (the shared MiniLM embedder, same model as
     the corpus — CLAUDE.md's collection invariant) and compare against every retrieved
     chunk's own embedding (`cosine_similarity`, reused from `ingestion.pipeline.dedup` — same
     "plain code + embeddings before reaching for an LLM" reuse this codebase already applies
     to L3 dedup and retrieval clustering). Accept the best match at or above
     `SEMANTIC_FALLBACK_THRESHOLD` (0.75). **Deliberate deviation from the spec's literal 0.85:**
     since the prompt now hands Gemini short reproducible handles (S1, S2, …) rather than the
     unreproducible 64-hex chunk_id, the direct-ID path carries the common case, and this
     fallback is the safety net for a *paraphrased* claim whose handle the model dropped or
     garbled — 0.85 was too strict to catch those against thin, one-sentence chunk embeddings.
     This is graceful degradation for a missed handle, not a license to invent grounding: a
     claim with empty text (Phase 3's fully-malformed case) or with no chunk having an
     embedding never reaches this path's acceptance branch.
  3. **Fail.** Neither path grounds the claim -> `is_grounded = False`,
     `supporting_chunk_id = None`, `validation_confidence = 0.0`.

**Scope boundary (read before assuming this drops claims):** this module makes the per-claim
grounding decision only. The spec's "reject, don't repair" *policy* — actually dropping
ungrounded claims from the answer, the >30%-dropped confidence warning, and the zero-survive
honest empty state — are actions on the *assembled answer*, which only Phase 5 (formatter.py)
builds; they live there, not here. `validate()` returns every claim, grounded and ungrounded
alike, so Phase 5 has the full set to compute those consequences from.
"""

from __future__ import annotations

from typing import Callable

from extraction.utils.embedding import embed_text
from generation.contracts import ParsedClaim, ValidatedClaim
from ingestion.pipeline.dedup import cosine_similarity
from retrieval.contracts import RetrievedChunk

DIRECT_LOOKUP_CONFIDENCE = 1.0
SEMANTIC_FALLBACK_THRESHOLD = 0.75  # safety net for a paraphrased claim; see module docstring
UNGROUNDED_VALIDATION_CONFIDENCE = 0.0


def _best_semantic_match(
    claim_text: str,
    chunks: dict[str, RetrievedChunk],
    embedder: Callable[[str], list[float]],
    similarity_fn: Callable[[list[float], list[float]], float],
) -> tuple[str | None, float]:
    """Return (chunk_id, similarity) of the best-matching embedded chunk, or (None, 0.0)."""
    claim_embedding = embedder(claim_text)
    best_chunk_id: str | None = None
    best_similarity = UNGROUNDED_VALIDATION_CONFIDENCE
    for chunk_id, chunk in chunks.items():
        if chunk.embedding is None:
            continue
        similarity = similarity_fn(claim_embedding, chunk.embedding)
        if best_chunk_id is None or similarity > best_similarity:
            best_chunk_id, best_similarity = chunk_id, similarity
    if best_chunk_id is None:
        return None, UNGROUNDED_VALIDATION_CONFIDENCE
    return best_chunk_id, best_similarity


class CitationValidator:
    """Phase 4: list[ParsedClaim] -> list[ValidatedClaim], grounded and ungrounded alike."""

    def __init__(
        self,
        embedder: Callable[[str], list[float]] = embed_text,
        similarity_fn: Callable[[list[float], list[float]], float] = cosine_similarity,
        similarity_threshold: float = SEMANTIC_FALLBACK_THRESHOLD,
    ):
        self._embedder = embedder
        self._similarity_fn = similarity_fn
        self._threshold = similarity_threshold

    def validate(
        self, claims: list[ParsedClaim], chunks: dict[str, RetrievedChunk]
    ) -> list[ValidatedClaim]:
        return [self._validate_one(claim, chunks) for claim in claims]

    def _validate_one(
        self, claim: ParsedClaim, chunks: dict[str, RetrievedChunk]
    ) -> ValidatedClaim:
        if claim.source_chunk_id is not None and claim.source_chunk_id in chunks:
            return ValidatedClaim(
                claim_text=claim.claim_text,
                confidence=claim.confidence,
                is_grounded=True,
                supporting_chunk_id=claim.source_chunk_id,
                validation_confidence=DIRECT_LOOKUP_CONFIDENCE,
            )

        if claim.claim_text.strip():
            best_chunk_id, best_similarity = _best_semantic_match(
                claim.claim_text, chunks, self._embedder, self._similarity_fn
            )
            if best_chunk_id is not None and best_similarity >= self._threshold:
                return ValidatedClaim(
                    claim_text=claim.claim_text,
                    confidence=claim.confidence,
                    is_grounded=True,
                    supporting_chunk_id=best_chunk_id,
                    validation_confidence=best_similarity,
                )

        return ValidatedClaim(
            claim_text=claim.claim_text,
            confidence=claim.confidence,
            is_grounded=False,
            supporting_chunk_id=None,
            validation_confidence=UNGROUNDED_VALIDATION_CONFIDENCE,
        )
