"""
Contract tests for the generation dataclasses (generation/contracts.py).

Pins immutability (frozen, mirroring retrieval.contracts's convention) and that
ParsedClaim's malformed-block representation (source_chunk_id/confidence as None,
is_valid=False) is expressible without a required str forcing a placeholder value.
"""

from __future__ import annotations

import pytest

from tests.generation.fixtures import (
    make_citation,
    make_generated_answer,
    make_lens_doc,
    make_parsed_claim,
    make_validated_claim,
)


class TestFrozenContracts:
    def test_lens_doc_is_frozen(self):
        doc = make_lens_doc()
        with pytest.raises(AttributeError):
            doc.title = "changed"

    def test_parsed_claim_is_frozen(self):
        claim = make_parsed_claim()
        with pytest.raises(AttributeError):
            claim.claim_text = "changed"

    def test_validated_claim_is_frozen(self):
        claim = make_validated_claim()
        with pytest.raises(AttributeError):
            claim.is_grounded = False

    def test_citation_is_frozen(self):
        citation = make_citation()
        with pytest.raises(AttributeError):
            citation.tier = 0

    def test_generated_answer_is_frozen(self):
        answer = make_generated_answer()
        with pytest.raises(AttributeError):
            answer.prose = "changed"


class TestParsedClaimMalformedRepresentation:
    def test_well_formed_block_defaults(self):
        claim = make_parsed_claim()
        assert claim.is_valid is True
        assert claim.source_chunk_id is not None
        assert claim.confidence is not None

    def test_malformed_block_can_omit_id_and_confidence(self):
        claim = make_parsed_claim(source_chunk_id=None, confidence=None, is_valid=False)
        assert claim.source_chunk_id is None
        assert claim.confidence is None
        assert claim.is_valid is False
