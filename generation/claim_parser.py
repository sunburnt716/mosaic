"""
Phase 3 — Claim Parsing: turn Gemini's structured text into typed ParsedClaim objects.

Deterministic, line-by-line parsing — no LLM re-parse, no freeform-prose handling. The format
is constrained (Phase 2's CLAIM/SOURCE_CHUNK_ID/CONFIDENCE contract), so a plain prefix match
per line is the whole parser.

A `---`-delimited segment is dropped only if it's pure whitespace (delimiter noise, not
content the model produced — e.g. a trailing `---` with nothing after it). Every other segment
becomes a ParsedClaim, even one with no recognizable CLAIM/SOURCE_CHUNK_ID lines at all: Phase 3
"passes forward marked invalid for Phase 4 to reject" rather than silently dropping malformed
or ID-less blocks, so a segment missing required fields still surfaces as
`ParsedClaim(is_valid=False, ...)` instead of vanishing.
"""

from __future__ import annotations

from generation.contracts import ParsedClaim

_DELIMITER = "---"
_CLAIM_PREFIX = "CLAIM:"
_SOURCE_CHUNK_ID_PREFIX = "SOURCE_CHUNK_ID:"
_CONFIDENCE_PREFIX = "CONFIDENCE:"


def _parse_block(segment: str) -> ParsedClaim:
    claim_text = ""
    source_chunk_id: str | None = None
    confidence: str | None = None

    for line in segment.splitlines():
        stripped = line.strip()
        if stripped.startswith(_CLAIM_PREFIX):
            claim_text = stripped[len(_CLAIM_PREFIX) :].strip()
        elif stripped.startswith(_SOURCE_CHUNK_ID_PREFIX):
            source_chunk_id = stripped[len(_SOURCE_CHUNK_ID_PREFIX) :].strip() or None
        elif stripped.startswith(_CONFIDENCE_PREFIX):
            confidence = stripped[len(_CONFIDENCE_PREFIX) :].strip() or None

    is_valid = bool(claim_text) and bool(source_chunk_id)
    return ParsedClaim(
        claim_text=claim_text,
        source_chunk_id=source_chunk_id,
        confidence=confidence,
        is_valid=is_valid,
    )


class ClaimParser:
    """Phase 3: Gemini's raw structured text -> list[ParsedClaim] (valid and invalid alike)."""

    def parse(self, raw_text: str) -> list[ParsedClaim]:
        return [_parse_block(segment) for segment in raw_text.split(_DELIMITER) if segment.strip()]
