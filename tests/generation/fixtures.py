"""
Shared, fully-offline builders for generation-layer tests.

Mirrors tests/retrieval/fixtures.py's convention: sane defaults so individual tests only
override what they care about. Not a test module (no `test_` functions), so pytest does not
collect it; test files import these builders directly.
"""

from __future__ import annotations

from generation.contracts import Citation, GeneratedAnswer, LensDoc, ParsedClaim, ValidatedClaim


def make_lens_doc(**overrides) -> LensDoc:
    base = dict(
        title="Framework: corroboration over conviction",
        text="Weigh claims by how many independent, credible sources corroborate them.",
    )
    base.update(overrides)
    return LensDoc(**base)


def make_parsed_claim(**overrides) -> ParsedClaim:
    base = dict(
        claim_text="NVIDIA beat Q2 earnings expectations.",
        source_chunk_id="doc-0001#0",
        confidence="high",
        is_valid=True,
    )
    base.update(overrides)
    return ParsedClaim(**base)


def make_validated_claim(**overrides) -> ValidatedClaim:
    base = dict(
        claim_text="NVIDIA beat Q2 earnings expectations.",
        confidence="high",
        is_grounded=True,
        supporting_chunk_id="doc-0001#0",
        validation_confidence=1.0,
    )
    base.update(overrides)
    return ValidatedClaim(**base)


def make_citation(**overrides) -> Citation:
    base = dict(
        text="NVIDIA beat Q2 earnings expectations.",
        url_with_fragment="https://example.com/article/0001#:~:text=NVIDIA%20beat",
        source="Reuters · Tier 1",
        tier=1,
    )
    base.update(overrides)
    return Citation(**base)


def make_generated_answer(**overrides) -> GeneratedAnswer:
    base = dict(
        prose="NVIDIA beat Q2 earnings expectations.",
        citations=[make_citation()],
        confidence_warning=None,
        corroboration_summary={},
    )
    base.update(overrides)
    return GeneratedAnswer(**base)
