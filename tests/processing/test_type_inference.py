"""
Tests for processing/type_inference.py.

Covers, generically across source shapes (not one source at a time):
  - inference accuracy per type (filing / article / tweet / unknown),
  - the structural sub-signals in isolation,
  - the advisory-vs-structure reconciliation policy (override and fallback).

Fully offline — bodies come from tests/processing/fixtures, no network or model.
"""

from __future__ import annotations

import pytest

from processing.type_inference import (
    ARTICLE,
    FILING,
    TWEET,
    UNKNOWN,
    _looks_like_article,
    _looks_like_filing,
    _looks_like_tweet,
    infer_document_type,
)
from tests.processing.fixtures import (
    AMBIGUOUS_BODY,
    ARTICLE_BODY,
    FILING_BODY,
    HEADERLESS_FILING_BODY,
    TWEET_BODY,
    make_document,
)

# ---------------------------------------------------------------------------
# Structural sub-signals (independently testable, per the design)
# ---------------------------------------------------------------------------


class TestSubSignals:
    def test_looks_like_filing_needs_markers_and_length(self):
        assert _looks_like_filing(token_count=600, marker_count=2) is True
        assert _looks_like_filing(token_count=600, marker_count=1) is False  # too few markers
        assert _looks_like_filing(token_count=400, marker_count=3) is False  # too short

    def test_looks_like_tweet_needs_short_single_block_no_markers(self):
        assert _looks_like_tweet(token_count=20, marker_count=0, paragraph_count=1) is True
        assert _looks_like_tweet(token_count=20, marker_count=1, paragraph_count=1) is False
        assert _looks_like_tweet(token_count=20, marker_count=0, paragraph_count=3) is False
        assert _looks_like_tweet(token_count=400, marker_count=0, paragraph_count=1) is False

    def test_looks_like_article_is_mid_length_without_filing_structure(self):
        assert _looks_like_article(token_count=300, marker_count=0) is True
        assert _looks_like_article(token_count=80, marker_count=0) is False  # too short
        assert _looks_like_article(token_count=6000, marker_count=0) is False  # too long
        assert _looks_like_article(token_count=300, marker_count=2) is False  # filing markers


# ---------------------------------------------------------------------------
# Inference accuracy
# ---------------------------------------------------------------------------


class TestInferenceAccuracy:
    def test_edgar_filing_infers_filing(self):
        doc = make_document(source_name="sec-edgar-8k", tier=0, body=FILING_BODY)
        assert infer_document_type(doc) == FILING

    def test_wire_article_infers_article(self):
        doc = make_document(source_name="reuters-business-rss", tier=1, body=ARTICLE_BODY)
        assert infer_document_type(doc) == ARTICLE

    def test_social_post_infers_tweet(self):
        doc = make_document(source_name="stocktwits", tier=3, body=TWEET_BODY)
        assert infer_document_type(doc) == TWEET

    def test_empty_body_infers_unknown(self):
        assert infer_document_type(make_document(body="")) == UNKNOWN
        assert infer_document_type(make_document(body="   \n\t  ")) == UNKNOWN

    def test_structurally_ambiguous_body_infers_unknown_without_advisory(self):
        doc = make_document(body=AMBIGUOUS_BODY)
        assert infer_document_type(doc) == UNKNOWN


# ---------------------------------------------------------------------------
# Advisory vs structure reconciliation
# ---------------------------------------------------------------------------


class TestAdvisoryReconciliation:
    def test_advisory_decides_when_structure_is_ambiguous(self):
        doc = make_document(source_name="some-feed", body=AMBIGUOUS_BODY)
        hints = {"some-feed": ARTICLE}
        assert infer_document_type(doc, source_hints=hints) == ARTICLE

    def test_strong_filing_structure_overrides_article_advisory(self):
        # Source is declared "article" but the body is unmistakably a filing.
        doc = make_document(source_name="mislabeled-feed", body=FILING_BODY)
        hints = {"mislabeled-feed": ARTICLE}
        assert infer_document_type(doc, source_hints=hints) == FILING

    def test_strong_tweet_structure_overrides_article_advisory(self):
        # Tweet sources can only be configured as "article" (no "tweet" in config
        # vocabulary), so a confidently tiny body must still infer as a tweet.
        doc = make_document(source_name="x-feed", body=TWEET_BODY)
        hints = {"x-feed": ARTICLE}
        assert infer_document_type(doc, source_hints=hints) == TWEET

    def test_filing_advisory_applies_to_headerless_filing(self):
        # Long, marker-free body: structure alone reads as an article, but the source
        # advisory (filing) wins because structure does not strongly disagree. The
        # missing headers are then surfaced by validation, not by re-typing here.
        doc = make_document(source_name="sec-edgar-8k", tier=0, body=HEADERLESS_FILING_BODY)
        hints = {"sec-edgar-8k": FILING}
        assert infer_document_type(doc, source_hints=hints) == FILING

    def test_out_of_vocabulary_or_unknown_source_hint_is_ignored(self):
        doc = make_document(source_name="feed", body=AMBIGUOUS_BODY)
        assert infer_document_type(doc, source_hints={"feed": "nonsense"}) == UNKNOWN
        assert infer_document_type(doc, source_hints={"other": ARTICLE}) == UNKNOWN


@pytest.mark.parametrize(
    "body, expected",
    [
        (FILING_BODY, FILING),
        (ARTICLE_BODY, ARTICLE),
        (TWEET_BODY, TWEET),
        (AMBIGUOUS_BODY, UNKNOWN),
    ],
)
def test_inference_is_deterministic_across_types(body, expected):
    """Same input → same output, with no advisory hint in play."""
    doc = make_document(body=body)
    assert infer_document_type(doc) == expected
