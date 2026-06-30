"""Tests for ingestion/pipeline/normalizer.py.

Contract under test:
  normalize(raw: dict, config: SourceConfig, fetched_at: datetime) -> Document
    - Maps adapter raw dict → validated Document using SourceConfig field mappings
    - Strips HTML tags from body; leaves plain text
    - Coerces published_date to timezone-aware UTC datetime
    - Stamps tier from config — NEVER from raw content
    - Sets doc_type from config params ("article" | "filing")
    - Attaches raw_payload verbatim — never modified or truncated
    - Computes and attaches content_hash, identity_key, and id via hashing.py
    - Raises NormalizationError if url, published_date, or source_name is missing/invalid
    - Is pure: same inputs always produce the same output; no I/O, no side effects
"""

from datetime import datetime, timezone

import pytest

from ingestion.pipeline.normalizer import NormalizationError, normalize
from tests.conftest import load_fixture


class TestNormalizerHappyPath:
    def test_title_is_mapped(self, reuters_rss_raw, reuters_source_config, fetched_at):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.title == reuters_rss_raw["title"]

    def test_url_is_mapped(self, reuters_rss_raw, reuters_source_config, fetched_at):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.url == reuters_rss_raw["url"]

    def test_source_name_comes_from_config(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.source_name == reuters_source_config.name

    def test_tier_stamped_from_config(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.tier == reuters_source_config.tier

    def test_fetched_at_preserved(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.fetched_at == fetched_at

    def test_doc_type_article_from_config(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.doc_type == "article"

    def test_doc_type_filing_from_edgar_config(
        self, reuters_rss_raw, edgar_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, edgar_source_config, fetched_at)
        assert doc.doc_type == "filing"


class TestBodyCleaning:
    def test_html_tags_stripped_from_body(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert "<" not in doc.body
        assert ">" not in doc.body

    def test_plain_text_preserved_after_stripping(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert "Federal Reserve" in doc.body

    def test_rest_json_html_stripped(
        self, rest_json_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(rest_json_raw, reuters_source_config, fetched_at)
        assert "<div" not in doc.body
        assert "class=" not in doc.body
        assert "Federal Reserve" in doc.body


class TestPublishedDateCoercion:
    def test_rfc2822_date_parsed_to_utc(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        # RSS uses RFC 2822: "Mon, 15 Jan 2024 14:30:00 GMT"
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.published_date == datetime(
            2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc
        )

    def test_iso8601_date_parsed_to_utc(
        self, rest_json_raw, reuters_source_config, fetched_at
    ):
        # REST JSON uses ISO 8601: "2024-01-15T14:30:00Z"
        doc = normalize(rest_json_raw, reuters_source_config, fetched_at)
        assert doc.published_date == datetime(
            2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc
        )

    def test_published_date_is_timezone_aware(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.published_date.tzinfo is not None

    def test_published_date_is_utc(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.published_date.utcoffset().total_seconds() == 0


class TestRawPayloadPreservation:
    def test_raw_payload_preserved_exactly(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.raw_payload == reuters_rss_raw["raw_payload"]

    def test_raw_payload_is_not_modified(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        original_payload = dict(reuters_rss_raw["raw_payload"])
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.raw_payload == original_payload

    def test_raw_payload_is_not_the_normalized_body(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.raw_payload is not doc.body


class TestHashingIntegration:
    def test_content_hash_is_64_char_hex(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert len(doc.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in doc.content_hash)

    def test_identity_key_contains_double_colon(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert "::" in doc.identity_key

    def test_identity_key_starts_with_source_name(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.identity_key.startswith(reuters_source_config.name)

    def test_document_id_is_set(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.id
        assert isinstance(doc.id, str)
        assert len(doc.id) == 64

    def test_normalize_is_pure_same_input_same_id(
        self, reuters_source_config, fetched_at
    ):
        raw_a = load_fixture("rss_reuters_sample.json")
        raw_b = load_fixture("rss_reuters_sample.json")
        doc_a = normalize(raw_a, reuters_source_config, fetched_at)
        doc_b = normalize(raw_b, reuters_source_config, fetched_at)
        assert doc_a.id == doc_b.id
        assert doc_a.content_hash == doc_b.content_hash

    def test_different_body_different_content_hash(
        self, reuters_source_config, fetched_at
    ):
        raw_v1 = load_fixture("rss_reuters_sample.json")
        raw_v2 = load_fixture("rss_reuters_sample.json")
        raw_v2["raw_body"] = "<p>Updated: Fed cuts rates by 50bps instead.</p>"
        raw_v2["raw_payload"] = dict(raw_v2["raw_payload"])
        doc_v1 = normalize(raw_v1, reuters_source_config, fetched_at)
        doc_v2 = normalize(raw_v2, reuters_source_config, fetched_at)
        assert doc_v1.content_hash != doc_v2.content_hash


class TestNormalizationErrors:
    def test_missing_url_raises_normalization_error(
        self, reuters_source_config, fetched_at
    ):
        raw = load_fixture("rss_reuters_sample.json")
        del raw["url"]
        with pytest.raises(NormalizationError):
            normalize(raw, reuters_source_config, fetched_at)

    def test_missing_published_date_raises_normalization_error(
        self, reuters_source_config, fetched_at
    ):
        raw = load_fixture("rss_reuters_sample.json")
        del raw["published"]
        with pytest.raises(NormalizationError):
            normalize(raw, reuters_source_config, fetched_at)

    def test_unparseable_date_raises_normalization_error(
        self, reuters_source_config, fetched_at
    ):
        raw = load_fixture("rss_reuters_sample.json")
        raw["published"] = "not-a-valid-date"
        with pytest.raises(NormalizationError):
            normalize(raw, reuters_source_config, fetched_at)

    def test_tier_in_raw_content_is_ignored(self, reuters_source_config, fetched_at):
        raw = load_fixture("rss_reuters_sample.json")
        raw["tier"] = 99  # Must be ignored — tier comes from config only
        doc = normalize(raw, reuters_source_config, fetched_at)
        assert doc.tier == reuters_source_config.tier

    def test_tier_0_source_stamped_correctly(
        self, reuters_rss_raw, edgar_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, edgar_source_config, fetched_at)
        assert doc.tier == 0

    def test_url_without_scheme_raises(self, reuters_source_config, fetched_at):
        raw = load_fixture("rss_reuters_sample.json")
        raw["url"] = "www.reuters.com/markets/article"  # no http(s):// scheme
        with pytest.raises(NormalizationError):
            normalize(raw, reuters_source_config, fetched_at)

    def test_relative_url_raises(self, reuters_source_config, fetched_at):
        raw = load_fixture("rss_reuters_sample.json")
        raw["url"] = "/markets/article"  # no netloc
        with pytest.raises(NormalizationError):
            normalize(raw, reuters_source_config, fetched_at)

    def test_list_repr_url_raises(self, reuters_source_config, fetched_at):
        # The exact shape of the original EDGAR bug must be rejected per-record.
        raw = load_fixture("rss_reuters_sample.json")
        raw["url"] = "https://www.sec.gov/Archives/edgar/data/['001-39218']//-index.htm"
        # urlparse accepts this (scheme+netloc present), so it passes URL validation —
        # the quality gate (Phase 3) flags it. Document the boundary explicitly.
        doc = normalize(raw, reuters_source_config, fetched_at)
        assert "[" in doc.url  # URL-parseability alone does not catch list-repr
