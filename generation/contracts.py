"""
Shared dataclasses passed between generation phases (Generation Pipeline spec).

Frozen, mirroring retrieval.contracts's convention: once a phase hands off its result,
downstream phases read it but never mutate it in place.

  LensDoc         — Phase 1 input: a permanent investing-framework doc used as prompt framing.
  ParsedClaim     — Phase 3 output: one CLAIM/SOURCE_CHUNK_ID/CONFIDENCE block, parsed.
  ValidatedClaim  — Phase 4 output: a claim after the grounding gate.
  Citation        — Phase 5 output element: one claim's deep-linked source.
  GeneratedAnswer — Phase 5 output: the final typed contract for the user-facing surface.

`LensDoc` is kept deliberately minimal (title + text): it's a data placeholder for real
investing-framework content that doesn't exist in this codebase yet — PromptBuilder accepts
whatever list it's given and ships no bundled content of its own.

`ParsedClaim.source_chunk_id`/`.confidence` are `| None` beyond the spec's literal 3-field
listing: Phase 3's own logic requires "malformed or ID-less blocks are passed forward marked
invalid for Phase 4 to reject" — that needs *some* way to represent "this block didn't parse
a usable ID/confidence," which a required `str` can't express. `is_valid` is the explicit,
unambiguous flag Phase 4 checks; a missing ID alone would have been an implicit, easy-to-miss
proxy for the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LensDoc:
    """A permanent investing-framework doc included as prompt framing, not prescription."""

    title: str
    text: str


@dataclass(frozen=True)
class ParsedClaim:
    """Phase 3 output: one structured block from Gemini's synthesis output."""

    claim_text: str
    source_chunk_id: str | None  # None if the block had no usable chunk ID
    confidence: str | None  # "high" | "medium" | "low"; None if the block omitted it
    is_valid: bool = True  # False for a malformed/ID-less block; Phase 4 must reject these


@dataclass(frozen=True)
class ValidatedClaim:
    """Phase 4 output: a claim after the grounding gate."""

    claim_text: str
    confidence: str | None
    is_grounded: bool
    supporting_chunk_id: str | None
    validation_confidence: float


@dataclass(frozen=True)
class Citation:
    """Phase 5 output element: one grounded claim's deep-linked, tier-labeled source.

    `source` is a formatted label, e.g. "Reuters · Tier 1" — for Tier 2/3 it also carries the
    skepticism note inline (e.g. "Blog · Tier 3 (read with appropriate skepticism)"), rather
    than a separate boolean flag, so the contract matches the spec's literal
    {text, url_with_fragment, source, tier} shape exactly.
    """

    text: str  # the exact sentence selected from the supporting chunk
    url_with_fragment: str  # deep link, e.g. "https://example.com/a#:~:text=..."
    source: str
    tier: int


@dataclass(frozen=True)
class GeneratedAnswer:
    """Phase 5 output: the final typed contract for the user-facing surface."""

    prose: str  # claims woven into readable text
    citations: list[Citation]
    confidence_warning: str | None
    corroboration_summary: dict[str, int]  # e.g. {"earnings_beat": 4} (outlets per story)
