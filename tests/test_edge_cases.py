"""Edge-case tests for the ingestion pipeline — the suite CI leans on to catch
regressions in the tricky parts: hash stability, HTML/date coercion, dedup
priority ordering, and adapter error isolation.

These complement the per-module contract tests. Grouped by stage for readability.
"""

from datetime import datetime, timezone

import pytest

from ingestion.pipeline.dedup import (
    L3_SIMILARITY_THRESHOLD,
    DedupResult,
    classify,
)
from ingestion.pipeline.hashing import content_hash, document_id, identity_key
from ingestion.pipeline.normalizer import NormalizationError, normalize
from tests.conftest import load_fixture, make_document, make_source_config
from tests.test_dedup import MockSeenStore

# ---------------------------------------------------------------------------
# Hashing edge cases
# ---------------------------------------------------------------------------


class TestHashingEdgeCases:
    def test_empty_string_hashes_to_valid_hex(self):
        result = content_hash("")
        assert len(result) == 64

    def test_whitespace_only_equals_empty(self):
        # Both normalize to the empty string before hashing.
        assert content_hash("   \n\t  ") == content_hash("")

    def test_unicode_str_and_utf8_bytes_match(self):
        text = "Fed raises rates — markets tumble 📉 café"
        assert content_hash(text) == content_hash(text.encode("utf-8"))

    def test_unicode_is_stable(self):
        assert content_hash("café ☕") == content_hash("café ☕")

    def test_very_long_content_hashes(self):
        big = "rate cut " * 100_000
        assert len(content_hash(big)) == 64

    def test_document_id_separator_prevents_concatenation_collision(self):
        # Without a separator, ("ab","c") and ("a","bc") would both hash "abc".
        assert document_id("ab", "c") != document_id("a", "bc")

    def test_identity_key_preserves_colons_in_article_id(self):
        # EDGAR accession numbers and RSS guids can contain colons.
        key = identity_key("Reuters", "tag:reuters.com,2024:newsml_X")
        assert key == "Reuters::tag:reuters.com,2024:newsml_X"


# ---------------------------------------------------------------------------
# Normalizer edge cases
# ---------------------------------------------------------------------------


class TestNormalizerEdgeCases:
    def _raw(self, **overrides):
        raw = load_fixture("rss_reuters_sample.json")
        raw.update(overrides)
        return raw

    def test_nested_html_fully_stripped(self, reuters_source_config, fetched_at):
        raw = self._raw(
            raw_body='<div class="x"><p>Fed <b>cuts</b> <a href="#">rates</a></p></div>'
        )
        doc = normalize(raw, reuters_source_config, fetched_at)
        assert "<" not in doc.body and ">" not in doc.body
        assert "Fed cuts rates" in doc.body

    def test_html_entities_decoded(self, reuters_source_config, fetched_at):
        raw = self._raw(raw_body="<p>AT&amp;T raises guidance &lt;Q3&gt;</p>")
        doc = normalize(raw, reuters_source_config, fetched_at)
        assert "AT&T" in doc.body
        assert "&amp;" not in doc.body

    def test_timezone_offset_converted_to_utc(self, reuters_source_config, fetched_at):
        raw = self._raw(published="2024-01-15T09:30:00-05:00")
        doc = normalize(raw, reuters_source_config, fetched_at)
        assert doc.published_date == datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

    def test_doc_type_defaults_to_article_when_absent(self, fetched_at):
        config = make_source_config(params={})  # no doc_type key
        doc = normalize(load_fixture("rss_reuters_sample.json"), config, fetched_at)
        assert doc.doc_type == "article"

    def test_empty_body_still_produces_valid_document(
        self, reuters_source_config, fetched_at
    ):
        raw = self._raw(raw_body="")
        doc = normalize(raw, reuters_source_config, fetched_at)
        assert doc.body == ""
        assert len(doc.content_hash) == 64

    def test_blank_url_string_rejected(self, reuters_source_config, fetched_at):
        raw = self._raw(url="")
        with pytest.raises(NormalizationError):
            normalize(raw, reuters_source_config, fetched_at)

    def test_naive_iso_timestamp_assumed_utc(self, reuters_source_config, fetched_at):
        raw = self._raw(published="2024-01-15T14:30:00")  # no tz
        doc = normalize(raw, reuters_source_config, fetched_at)
        assert doc.published_date == datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Dedup priority + threshold edge cases
# ---------------------------------------------------------------------------


class TestDedupPriorityEdgeCases:
    def test_l1_beats_l3(self):
        # Same content hash AND a near-identical embedding -> L1 (hash checked first).
        store = MockSeenStore()
        store.add(
            make_document(content_hash="h1", identity_key="Bloomberg::a"),
            embedding=[1.0, 0.0],
        )
        incoming = make_document(content_hash="h1", identity_key="Reuters::b")
        assert (
            classify(incoming, store, embedding=[1.0, 0.0]) == DedupResult.L1_DUPLICATE
        )

    def test_l2_beats_l3(self):
        # Same identity (diff hash) AND a near embedding -> L2 (identity before embedding).
        store = MockSeenStore()
        store.add(
            make_document(content_hash="h1", identity_key="Reuters::a"),
            embedding=[1.0, 0.0],
        )
        incoming = make_document(content_hash="h2", identity_key="Reuters::a")
        assert classify(incoming, store, embedding=[1.0, 0.0]) == DedupResult.L2_UPDATE

    def test_similarity_just_above_threshold_is_l3(self):
        # cos(30°) ≈ 0.866 > 0.85
        store = MockSeenStore()
        store.add(make_document(identity_key="Bloomberg::a"), embedding=[1.0, 0.0])
        incoming = make_document(identity_key="Reuters::b", content_hash="other")
        result = classify(incoming, store, embedding=[0.8660254, 0.5])
        assert result == DedupResult.L3_NEAR_DUPLICATE

    def test_similarity_just_below_threshold_is_new(self):
        # cos(45°) ≈ 0.707 < 0.85
        store = MockSeenStore()
        store.add(make_document(identity_key="Bloomberg::a"), embedding=[1.0, 0.0])
        incoming = make_document(identity_key="Reuters::b", content_hash="other")
        result = classify(incoming, store, embedding=[0.7071, 0.7071])
        assert result == DedupResult.NEW

    def test_threshold_constant_is_sane(self):
        assert 0.0 < L3_SIMILARITY_THRESHOLD < 1.0

    def test_embedding_provided_but_store_empty_is_new(self):
        store = MockSeenStore()
        assert classify(make_document(), store, embedding=[1.0, 0.0]) == DedupResult.NEW

    def test_one_of_many_embeddings_matches(self):
        store = MockSeenStore()
        store.add(make_document(identity_key="A::1"), embedding=[0.0, 1.0])
        store.add(make_document(identity_key="B::2"), embedding=[1.0, 0.0])
        incoming = make_document(identity_key="C::3", content_hash="other")
        assert (
            classify(incoming, store, embedding=[1.0, 0.0])
            == DedupResult.L3_NEAR_DUPLICATE
        )


# ---------------------------------------------------------------------------
# Adapter error-isolation + laziness edge cases
# ---------------------------------------------------------------------------


class TestAdapterEdgeCases:
    def test_empty_feed_yields_nothing_without_error(
        self, monkeypatch, reuters_source_config
    ):
        from ingestion.adapters.rss import RssAdapter

        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [])
        assert list(adapter.fetch(reuters_source_config)) == []

    def test_fetch_returns_lazy_generator(self, monkeypatch, reuters_source_config):
        from ingestion.adapters.rss import RssAdapter

        adapter = RssAdapter()
        called = {"hit": False}

        def spy(url, headers):
            called["hit"] = True
            return [load_fixture("rss_reuters_sample.json")]

        monkeypatch.setattr(adapter, "_fetch_feed", spy)
        gen = adapter.fetch(reuters_source_config)
        # Nothing fetched until the generator is iterated.
        assert called["hit"] is False
        next(gen)
        assert called["hit"] is True

    def test_fetch_error_chains_original_cause(
        self, monkeypatch, reuters_source_config
    ):
        from ingestion.adapters.base import FetchError
        from ingestion.adapters.rss import RssAdapter

        adapter = RssAdapter()

        def boom(url, headers):
            raise ConnectionError("DNS failure")

        monkeypatch.setattr(adapter, "_fetch_feed", boom)
        with pytest.raises(FetchError) as exc_info:
            list(adapter.fetch(reuters_source_config))
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_multiple_items_each_preserve_raw_payload(
        self, monkeypatch, reuters_source_config
    ):
        from ingestion.adapters.rss import RssAdapter

        raw_a = load_fixture("rss_reuters_sample.json")
        raw_b = load_fixture("rss_reuters_sample.json")
        raw_b["raw_payload"] = {"id": "second", "title": "Another"}
        adapter = RssAdapter()
        monkeypatch.setattr(adapter, "_fetch_feed", lambda url, headers: [raw_a, raw_b])

        results = list(adapter.fetch(reuters_source_config))
        assert results[0]["raw_payload"] == raw_a["raw_payload"]
        assert results[1]["raw_payload"] == raw_b["raw_payload"]
