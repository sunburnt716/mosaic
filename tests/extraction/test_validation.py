"""
Tests for processing/validation.py.

Covers that each per-type gate fires correctly, that severity and is_valid track the
result model, and that validation never raises and never blocks. Fully offline.
"""

from __future__ import annotations

from extraction.type_inference import ARTICLE, FILING, TWEET, UNKNOWN
from extraction.validation import (
    DEGENERATE,
    INFO,
    WARNING,
    validate_document,
)
from tests.extraction.fixtures import (
    ARTICLE_BODY,
    FILING_BODY,
    HEADERLESS_FILING_BODY,
    OVERSIZED_TWEET_BODY,
    SHORT_ARTICLE_BODY,
    TWEET_BODY,
    make_document,
)

# ---------------------------------------------------------------------------
# Filing validation
# ---------------------------------------------------------------------------


class TestFilingValidation:
    def test_well_formed_filing_is_clean(self):
        result = validate_document(make_document(body=FILING_BODY), FILING)
        assert result.is_valid is True
        assert result.severity == INFO
        assert result.warnings == []

    def test_headerless_filing_is_degenerate(self):
        # The signature degenerate case: typed filing, zero section markers.
        result = validate_document(make_document(body=HEADERLESS_FILING_BODY), FILING)
        assert result.severity == DEGENERATE
        assert result.is_valid is False
        assert any("zero section headers" in w for w in result.warnings)

    def test_short_filing_warns_but_stays_valid(self):
        # Has markers but is far shorter than a real filing.
        body = "Item 1.01 Entry into a Material Definitive Agreement. Risk Factors apply."
        result = validate_document(make_document(body=body), FILING)
        assert result.severity == WARNING
        assert result.is_valid is True
        assert result.warnings


# ---------------------------------------------------------------------------
# Article validation
# ---------------------------------------------------------------------------


class TestArticleValidation:
    def test_well_formed_article_is_clean(self):
        result = validate_document(make_document(body=ARTICLE_BODY), ARTICLE)
        assert result.severity == INFO
        assert result.is_valid is True
        assert result.warnings == []

    def test_tiny_article_is_degenerate(self):
        # A ~30-token "article" — too short to be meaningful prose.
        result = validate_document(make_document(body=SHORT_ARTICLE_BODY), ARTICLE)
        assert result.severity == DEGENERATE
        assert result.is_valid is False
        assert any("too short" in w for w in result.warnings)

    def test_oversized_article_warns(self):
        body = " ".join(["word"] * 6000)  # within prose but past the 5000 ceiling
        result = validate_document(make_document(body=body), ARTICLE)
        assert result.severity == WARNING
        assert result.is_valid is True
        assert any("outside the expected" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tweet validation
# ---------------------------------------------------------------------------


class TestTweetValidation:
    def test_well_formed_tweet_is_clean(self):
        result = validate_document(make_document(body=TWEET_BODY), TWEET)
        assert result.severity == INFO
        assert result.is_valid is True

    def test_oversized_tweet_warns(self):
        result = validate_document(make_document(body=OVERSIZED_TWEET_BODY), TWEET)
        assert result.severity == WARNING
        assert result.is_valid is True
        assert any("unexpectedly long" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Unknown / deferred and general guarantees
# ---------------------------------------------------------------------------


class TestUnknownAndGuarantees:
    def test_unknown_is_deferred_not_blocked(self):
        result = validate_document(make_document(body=ARTICLE_BODY), UNKNOWN)
        assert result.severity == WARNING
        assert result.is_valid is True  # deferred, not invalid
        assert any("validation_deferred" in w for w in result.warnings)

    def test_unrecognized_type_is_deferred_not_raised(self):
        result = validate_document(make_document(body=ARTICLE_BODY), "bogus-type")
        assert result.severity == WARNING
        assert result.is_valid is True
        assert result.warnings

    def test_validation_never_raises_on_empty_body(self):
        for inferred in (FILING, ARTICLE, TWEET, UNKNOWN):
            result = validate_document(make_document(body=""), inferred)
            # Returns a result for every type; never throws.
            assert isinstance(result.warnings, list)
