"""
Phase 1 — Prompt Assembly: build the Gemini prompt from the retrieved set + query + lens.

The prompt has four parts, in order: guardrails (system framing), the investing lens
(framing, never prescriptions), the output-format contract Gemini must follow (Phase 2's
CLAIM/SOURCE_CHUNK_ID/CONFIDENCE block format), and the labeled source blocks themselves.

Cluster selection is "corroboration strength × relevance" (spec's own phrase): each
StoryCluster's corroboration label maps to a strength constant, multiplied by its best chunk's
similarity_score, and the top ~5 clusters by that product are kept. Their chunks are flattened
in cluster-then-within-cluster order (already tier/recency-sorted by Phase 4) into one ranked
list, so "lowest-ranked chunks dropped first" under the token budget falls out naturally —
whichever chunks land at the tail of that list are the ones cut, never a mid-list truncation.

Token budget uses `text_metrics.count_tokens` (the cheap whitespace proxy already used by
Phase 0 type inference) rather than the real MiniLM tokenizer — this step doesn't feed an
embedder, so the heavier, network-dependent tokenizer isn't warranted here.

A dropped chunk is dropped whole — its SOURCE/CHUNK_ID/SECTION/TEXT block is all-or-nothing,
so citation metadata for an *included* chunk is never partially truncated (spec's "never drop
citation metadata" non-goal).
"""

from __future__ import annotations

from datetime import datetime, timezone

from generation.contracts import LensDoc
from extraction.text_metrics import count_tokens
from retrieval.contracts import RetrievedChunk, StoryCluster, UserProfile
from retrieval.output import RetrievalOutput

DEFAULT_TOKEN_BUDGET = 1500
DEFAULT_TOP_CLUSTERS = 5

# "Corroboration strength" — a cluster's corroboration label mapped to a numeric weight for
# ranking which stories make the prompt. Not a re-ranking score (that's retrieval's job); this
# only decides prompt inclusion order.
CORROBORATION_STRENGTH = {"high": 3.0, "medium": 2.0, "single": 1.0}

GUARDRAILS = (
    "You inform; you do not advise. Never issue a recommendation, price target, or "
    '"buy"/"sell"/"hold" call.',
    "Use only the provided sources below. Do not add outside facts or rely on prior knowledge.",
    "Flag Tier 2 and Tier 3 sources so the user can weigh their credibility.",
)

FORMAT_CONTRACT = """OUTPUT FORMAT (follow exactly, no prose outside this format):
CLAIM: <one factual claim, stated plainly>
SOURCE_CHUNK_ID: <the CHUNK_ID of the source block that supports this claim>
CONFIDENCE: high | medium | low
---
(repeat the CLAIM/SOURCE_CHUNK_ID/CONFIDENCE/--- block for each claim)"""


def _corroboration_rank(cluster: StoryCluster) -> float:
    if not cluster.chunks:
        return 0.0
    best_relevance = max(chunk.similarity_score for chunk in cluster.chunks)
    strength = CORROBORATION_STRENGTH.get(cluster.corroboration, 0.0)
    return strength * best_relevance


def _select_top_clusters(clusters: list[StoryCluster], top_n: int) -> list[StoryCluster]:
    return sorted(clusters, key=_corroboration_rank, reverse=True)[:top_n]


def _flatten_chunks(clusters: list[StoryCluster]) -> list[RetrievedChunk]:
    return [chunk for cluster in clusters for chunk in cluster.chunks]


def _render_chunk_block(chunk: RetrievedChunk) -> str:
    published = _format_epoch(chunk.published_epoch)
    section = chunk.section_label or "n/a"
    return (
        f"SOURCE: {chunk.source_name} (Tier {chunk.tier}) | Published: {published}\n"
        f"CHUNK_ID: {chunk.chunk_id}\n"
        f"SECTION: {section}\n"
        f"TEXT: {chunk.text}"
    )


def _format_epoch(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).date().isoformat()


def _render_lens(lens: list[LensDoc]) -> str:
    if not lens:
        return ""
    docs = "\n\n".join(f"- {doc.title}: {doc.text}" for doc in lens)
    return (
        "INVESTING FRAMEWORK (context for how to weigh sources — not instructions to "
        f"follow prescriptively):\n{docs}"
    )


def _render_profile(profile: UserProfile) -> str:
    interests = [*profile.tickers, *profile.sectors]
    if not interests:
        return ""
    return f"USER INTERESTS (for context only, not a filter): {', '.join(interests)}"


class PromptBuilder:
    """Phase 1: RetrievalOutput + query + lens + profile -> assembled Gemini prompt string."""

    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        top_clusters: int = DEFAULT_TOP_CLUSTERS,
    ):
        self._token_budget = token_budget
        self._top_clusters = top_clusters

    def build(
        self,
        retrieval: RetrievalOutput,
        query: str,
        lens: list[LensDoc],
        profile: UserProfile,
    ) -> str:
        selected_clusters = _select_top_clusters(retrieval.clusters, self._top_clusters)
        ranked_chunks = _flatten_chunks(selected_clusters)

        header_parts = [
            "\n".join(f"- {rule}" for rule in GUARDRAILS),
            _render_lens(lens),
            _render_profile(profile),
            FORMAT_CONTRACT,
            f"QUESTION: {query}",
            "SOURCES:",
        ]
        header = "\n\n".join(part for part in header_parts if part)
        remaining_budget = self._token_budget - count_tokens(header)

        included_blocks: list[str] = []
        for chunk in ranked_chunks:
            block = _render_chunk_block(chunk)
            block_tokens = count_tokens(block)
            if block_tokens > remaining_budget:
                break  # this and every lower-ranked chunk after it are dropped
            included_blocks.append(block)
            remaining_budget -= block_tokens

        return header + "\n\n" + "\n\n".join(included_blocks)
