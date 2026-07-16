"""
Contract + adversarial tests for Phase 3 Claim Parsing (generation/claim_parser.py).

Covers well-formed multi-claim text, malformed/ID-less blocks (must be passed forward marked
invalid, never silently dropped), delimiter/whitespace edge cases, and hostile Gemini output.
"""

from __future__ import annotations

from generation.claim_parser import ClaimParser


class TestWellFormedBlocks:
    def test_single_claim(self):
        text = "CLAIM: NVIDIA beat estimates.\nSOURCE_CHUNK_ID: doc-1#0\nCONFIDENCE: high\n---"
        claims = ClaimParser().parse(text)
        assert len(claims) == 1
        assert claims[0].claim_text == "NVIDIA beat estimates."
        assert claims[0].source_chunk_id == "doc-1#0"
        assert claims[0].confidence == "high"
        assert claims[0].is_valid is True

    def test_multiple_claims(self):
        text = (
            "CLAIM: First claim.\nSOURCE_CHUNK_ID: a#0\nCONFIDENCE: high\n"
            "---\n"
            "CLAIM: Second claim.\nSOURCE_CHUNK_ID: b#1\nCONFIDENCE: medium\n"
            "---"
        )
        claims = ClaimParser().parse(text)
        assert len(claims) == 2
        assert claims[0].claim_text == "First claim."
        assert claims[1].claim_text == "Second claim."
        assert all(c.is_valid for c in claims)

    def test_trailing_delimiter_not_required(self):
        text = "CLAIM: Only claim.\nSOURCE_CHUNK_ID: a#0\nCONFIDENCE: low"
        claims = ClaimParser().parse(text)
        assert len(claims) == 1
        assert claims[0].claim_text == "Only claim."

    def test_extra_whitespace_and_blank_lines_tolerated(self):
        text = (
            "\n\n  CLAIM: Spaced out.  \n\n  SOURCE_CHUNK_ID: a#0  \n\n"
            "  CONFIDENCE: high  \n\n---\n\n"
        )
        claims = ClaimParser().parse(text)
        assert len(claims) == 1
        assert claims[0].claim_text == "Spaced out."
        assert claims[0].source_chunk_id == "a#0"


class TestMalformedBlocksPassedForwardInvalid:
    def test_missing_source_chunk_id_marked_invalid_not_dropped(self):
        text = "CLAIM: Ungrounded-looking claim.\nCONFIDENCE: high\n---"
        claims = ClaimParser().parse(text)
        assert len(claims) == 1
        assert claims[0].is_valid is False
        assert claims[0].source_chunk_id is None
        assert claims[0].claim_text == "Ungrounded-looking claim."

    def test_missing_claim_text_marked_invalid_not_dropped(self):
        text = "SOURCE_CHUNK_ID: a#0\nCONFIDENCE: high\n---"
        claims = ClaimParser().parse(text)
        assert len(claims) == 1
        assert claims[0].is_valid is False
        assert claims[0].claim_text == ""

    def test_missing_confidence_still_valid_if_claim_and_id_present(self):
        # Confidence is a display detail, not required for Phase 4 grounding.
        text = "CLAIM: A claim.\nSOURCE_CHUNK_ID: a#0\n---"
        claims = ClaimParser().parse(text)
        assert claims[0].is_valid is True
        assert claims[0].confidence is None

    def test_empty_source_chunk_id_value_treated_as_missing(self):
        text = "CLAIM: A claim.\nSOURCE_CHUNK_ID:\nCONFIDENCE: high\n---"
        claims = ClaimParser().parse(text)
        assert claims[0].source_chunk_id is None
        assert claims[0].is_valid is False

    def test_completely_unrecognizable_block_still_surfaces_as_invalid(self):
        text = "The model just wrote some prose here instead of following the format.\n---"
        claims = ClaimParser().parse(text)
        assert len(claims) == 1
        assert claims[0].is_valid is False
        assert claims[0].claim_text == ""
        assert claims[0].source_chunk_id is None

    def test_multiple_blocks_mixed_valid_and_invalid(self):
        text = (
            "CLAIM: Good claim.\nSOURCE_CHUNK_ID: a#0\nCONFIDENCE: high\n"
            "---\n"
            "CLAIM: Bad claim, no id.\nCONFIDENCE: low\n"
            "---\n"
            "CLAIM: Another good one.\nSOURCE_CHUNK_ID: c#2\nCONFIDENCE: medium\n"
            "---"
        )
        claims = ClaimParser().parse(text)
        assert [c.is_valid for c in claims] == [True, False, True]


class TestDelimiterAndWhitespaceEdgeCases:
    def test_empty_string_yields_no_claims(self):
        assert ClaimParser().parse("") == []

    def test_whitespace_only_yields_no_claims(self):
        assert ClaimParser().parse("   \n\n   ") == []

    def test_only_delimiters_yields_no_claims(self):
        assert ClaimParser().parse("---\n---\n---") == []

    def test_consecutive_delimiters_produce_no_empty_claims(self):
        text = "CLAIM: One.\nSOURCE_CHUNK_ID: a#0\n------CLAIM: Two.\nSOURCE_CHUNK_ID: b#1\n---"
        claims = ClaimParser().parse(text)
        # "------" splits into an extra empty segment between the two real blocks, which
        # must be filtered out rather than becoming a third, invalid claim.
        assert len(claims) == 2

    def test_leading_and_trailing_whitespace_around_whole_response(self):
        text = "\n\n  CLAIM: Padded.\nSOURCE_CHUNK_ID: a#0\n---  \n\n"
        claims = ClaimParser().parse(text)
        assert len(claims) == 1
        assert claims[0].claim_text == "Padded."


class TestHostileGeminiOutput:
    def test_lowercase_labels_not_recognized(self):
        # The format contract specifies exact uppercase labels; a model that ignores
        # instructions and lowercases them produces an unrecognized (invalid) block rather
        # than a silent misparse.
        text = "claim: lowercase claim\nsource_chunk_id: a#0\n---"
        claims = ClaimParser().parse(text)
        assert claims[0].is_valid is False
        assert claims[0].claim_text == ""

    def test_duplicate_label_in_one_block_last_value_wins(self):
        text = "CLAIM: First.\nCLAIM: Second.\nSOURCE_CHUNK_ID: a#0\n---"
        claims = ClaimParser().parse(text)
        assert claims[0].claim_text == "Second."

    def test_confidence_value_outside_enum_passed_through_unvalidated(self):
        # ClaimParser extracts fields; it doesn't police the confidence vocabulary.
        text = "CLAIM: A claim.\nSOURCE_CHUNK_ID: a#0\nCONFIDENCE: extremely high\n---"
        claims = ClaimParser().parse(text)
        assert claims[0].confidence == "extremely high"

    def test_claim_text_containing_the_word_source_chunk_id(self):
        text = "CLAIM: The SOURCE_CHUNK_ID field matters here.\nSOURCE_CHUNK_ID: a#0\n---"
        claims = ClaimParser().parse(text)
        assert claims[0].claim_text == "The SOURCE_CHUNK_ID field matters here."
        assert claims[0].source_chunk_id == "a#0"

    def test_insufficient_data_marker_yields_no_valid_claims(self):
        # The marker has no "---" delimiter and no CLAIM:/SOURCE_CHUNK_ID: lines, so it
        # becomes a single invalid block rather than a genuinely empty list — either way,
        # Phase 4 sees zero *grounded* claims and falls into the "zero survive" honest
        # empty-state path, so the end behavior is the same.
        from generation.synthesizer import INSUFFICIENT_DATA_MARKER

        claims = ClaimParser().parse(INSUFFICIENT_DATA_MARKER)
        assert all(not c.is_valid for c in claims)

    def test_very_many_claims_does_not_crash(self):
        block = "CLAIM: X.\nSOURCE_CHUNK_ID: a#0\nCONFIDENCE: high\n---\n"
        text = block * 1000
        claims = ClaimParser().parse(text)
        assert len(claims) == 1000
        assert all(c.is_valid for c in claims)
