"""
Phase 5 — Output Formatting & Deep Links: assemble the final GeneratedAnswer.

**This is also where the spec's Phase 4 "reject, don't repair" rejection policy actually
executes** (see validator.py's module docstring for why the split is here, not there):
ungrounded claims are dropped, a >30%-dropped confidence warning is attached, and a
zero-survivor set returns the honest empty state rather than any answer at all. Never
backfills from the model's parametric memory — an ungrounded claim is simply absent, not
replaced with something plausible-sounding.

**Citation sentence selection is deterministic, never model-emitted.** For each grounded
claim, the supporting chunk's text is split into sentences (`text_metrics.sentence_spans` —
the same boundary definition Phase 1 chunking's highlight-span selection uses) and scored
against the claim by word overlap (Jaccard over lowercased word sets); the highest-scoring
sentence becomes the citation. This is a plain, explainable heuristic — not an embedding
call — because this phase's whole point is that nothing here can hallucinate a quote.

**Deep links** are a Chrome/Firefox text-fragment URL (`#:~:text=<quoted sentence>`) built
from the chunk's real `url` and the selected sentence — never a model-emitted span.

**Source labels** carry tier inline (`"Reuters · Tier 1"`); Tier 2/3 sources get an appended
skepticism note in that same string, per the spec's `Citation` shape
`{text, url_with_fragment, source, tier}` (no separate boolean flag — see contracts.py).

**`corroboration_summary`** is keyed by `StoryCluster.cluster_id` (a real, stable
identifier — the primary chunk's document id), not a human-readable topic label like the
spec's illustrative `"earnings_beat"`: generating an actual topic label would need
summarization or an LLM call, which is out of scope for a phase whose entire purpose is
avoiding exactly that kind of model-introduced content. Only clusters with at least one
*surviving, grounded* citation are included, mapped to that cluster's outlet_count.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from generation.contracts import Citation, GeneratedAnswer, ValidatedClaim
from extraction.text_metrics import sentence_spans
from retrieval.contracts import RetrievedChunk, StoryCluster

CONFIDENCE_WARNING_THRESHOLD = 0.30  # >30% dropped triggers the warning
CONFIDENCE_WARNING_MESSAGE = (
    "Limited corroborating sources for this query; this answer may be incomplete."
)
EMPTY_STATE_MESSAGE = "I don't have enough sourced information to answer that yet."

TIER_SKEPTICISM_THRESHOLD = 2  # Tier >= this gets the skepticism note
SKEPTICISM_NOTE = "read with appropriate skepticism"

_WORD_RE = re.compile(r"\w+")


def _words(text: str) -> set[str]:
    """Lowercased word set, punctuation stripped, for word-overlap scoring."""
    return set(_WORD_RE.findall(text.lower()))


def _select_best_sentence(claim_text: str, chunk_text: str) -> str:
    """Return the sentence in `chunk_text` with the highest word overlap with `claim_text`.

    Falls back to the first sentence (or the whole trimmed text, if no sentence boundary
    exists) when there's no overlap at all or nothing to score against — always returns
    *something* from the real chunk text, never an empty citation.
    """
    spans = sentence_spans(chunk_text)
    if not spans:
        return chunk_text.strip()

    claim_words = _words(claim_text)
    best_start, best_end = spans[0]
    best_score = -1.0
    for start, end in spans:
        sentence_words = _words(chunk_text[start:end])
        union = claim_words | sentence_words
        score = len(claim_words & sentence_words) / len(union) if union else 0.0
        if score > best_score:
            best_score, best_start, best_end = score, start, end
    return chunk_text[best_start:best_end]


def _build_deep_link(chunk: RetrievedChunk, sentence: str) -> str:
    return f"{chunk.url}#:~:text={quote(sentence)}"


def _format_source_label(chunk: RetrievedChunk) -> str:
    label = f"{chunk.source_name} · Tier {chunk.tier}"
    if chunk.tier >= TIER_SKEPTICISM_THRESHOLD:
        label = f"{label} ({SKEPTICISM_NOTE})"
    return label


def _build_citation(claim: ValidatedClaim, chunk: RetrievedChunk) -> Citation:
    sentence = _select_best_sentence(claim.claim_text, chunk.text)
    return Citation(
        text=sentence,
        url_with_fragment=_build_deep_link(chunk, sentence),
        source=_format_source_label(chunk),
        tier=chunk.tier,
    )


def _corroboration_summary(
    grounded_claims: list[ValidatedClaim], clusters: list[StoryCluster]
) -> dict[str, int]:
    cluster_by_chunk_id = {
        chunk.chunk_id: cluster for cluster in clusters for chunk in cluster.chunks
    }
    summary: dict[str, int] = {}
    for claim in grounded_claims:
        cluster = cluster_by_chunk_id.get(claim.supporting_chunk_id)
        if cluster is not None:
            summary[cluster.cluster_id] = cluster.outlet_count
    return summary


class AnswerFormatter:
    """Phase 5: (validated claims, chunks, clusters) -> GeneratedAnswer.

    `clusters` is additive beyond the spec's literal Phase 5 input list
    (`validated_claims`, `chunks`) — necessary to compute `corroboration_summary` from
    retrieval's already-computed `StoryCluster.outlet_count` rather than reinventing
    grouping logic here (documented decision, same pattern as retrieval's
    `RetrievedChunk.embedding`).
    """

    def format(
        self,
        validated_claims: list[ValidatedClaim],
        chunks: dict[str, RetrievedChunk],
        clusters: list[StoryCluster],
    ) -> GeneratedAnswer:
        grounded = [claim for claim in validated_claims if claim.is_grounded]

        if not grounded:
            return GeneratedAnswer(
                prose=EMPTY_STATE_MESSAGE,
                citations=[],
                confidence_warning=None,
                corroboration_summary={},
            )

        citations: list[Citation] = []
        prose_sentences: list[str] = []
        surviving_claims: list[ValidatedClaim] = []
        for claim in grounded:
            chunk = chunks.get(claim.supporting_chunk_id)
            if chunk is None:
                continue  # defensive: shouldn't happen if `chunks` matches what validated
                # this claim, but a claim with no real chunk to cite can't
                # be woven into the answer.
            citations.append(_build_citation(claim, chunk))
            prose_sentences.append(claim.claim_text)
            surviving_claims.append(claim)

        if not surviving_claims:
            return GeneratedAnswer(
                prose=EMPTY_STATE_MESSAGE,
                citations=[],
                confidence_warning=None,
                corroboration_summary={},
            )

        total = len(validated_claims)
        drop_rate = (total - len(surviving_claims)) / total if total else 0.0
        confidence_warning = (
            CONFIDENCE_WARNING_MESSAGE if drop_rate > CONFIDENCE_WARNING_THRESHOLD else None
        )

        return GeneratedAnswer(
            prose=" ".join(prose_sentences),
            citations=citations,
            confidence_warning=confidence_warning,
            corroboration_summary=_corroboration_summary(surviving_claims, clusters),
        )
