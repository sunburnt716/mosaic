"""Tests for ingestion/pipeline/quality.py — the soft, batch-aware quality gate.

All offline. Uses make_document() / make_source_config() from conftest.py. The gate returns
a QualityReport(warnings, stats); it must never raise and never drop. Helpers below build
distinct-by-default batches so a test only triggers the flag it's exercising.
"""

from ingestion.pipeline.quality import (
    DEFAULT_MAX_FALLBACK_TITLE_RATE,
    MIN_BATCH_FOR_COLLAPSE,
    QualityReport,
    check,
)
from tests.conftest import make_document, make_source_config


def _config(expects: dict | None = None, **thresholds):
    return make_source_config(
        expects=expects or {"body": True, "title": True, "url": True}, **thresholds
    )


def _healthy_batch(n: int):
    """n distinct, well-formed docs — should trip no flag."""
    return [
        make_document(
            title=f"Real headline {i}",
            body=f"Substantive body number {i}.",
            url=f"https://example.com/article/{i}",
            identity_key=f"src::{i}",
            content_hash=f"{i:064d}",
        )
        for i in range(n)
    ]


def _warnings(report: QualityReport) -> str:
    return " ".join(report.warnings)


# ---------------------------------------------------------------------------
# Return shape + stats
# ---------------------------------------------------------------------------


class TestReportShape:
    def test_returns_quality_report(self):
        report = check(_healthy_batch(5), _config())
        assert isinstance(report, QualityReport)
        assert isinstance(report.warnings, list)
        assert isinstance(report.stats, dict)

    def test_healthy_batch_no_warnings(self):
        report = check(_healthy_batch(6), _config())
        assert report.warnings == []

    def test_stats_are_computed(self):
        report = check(_healthy_batch(4), _config())
        assert report.stats["records"] == 4
        assert report.stats["empty_body_rate"] == 0.0
        assert report.stats["fallback_title_rate"] == 0.0
        assert report.stats["unique_urls"] == 4
        assert report.stats["unique_identity_keys"] == 4
        assert report.stats["unique_content_hashes"] == 4

    def test_empty_batch_stats_minimal(self):
        report = check([], _config())
        assert report.stats == {"records": 0}
        assert report.warnings == []  # no min_records set -> silent


# ---------------------------------------------------------------------------
# TITLE_FALLBACK
# ---------------------------------------------------------------------------


class TestTitleFallback:
    def test_warns_on_mostly_unknown_titles(self):
        docs = _healthy_batch(4)
        for d in docs[:3]:
            d.title = "filing — Unknown"
        report = check(docs, _config())
        assert "TITLE_FALLBACK" in _warnings(report)

    def test_silent_on_real_titles(self):
        report = check(_healthy_batch(6), _config())
        assert "TITLE_FALLBACK" not in _warnings(report)

    def test_per_source_threshold_honored_when_stricter(self):
        # 1/4 = 0.25 fallback rate: below the 0.50 default (silent), but a stricter
        # per-source max of 0.10 should make it warn.
        docs = _healthy_batch(4)
        docs[0].title = "Untitled"
        assert "TITLE_FALLBACK" not in _warnings(check(docs, _config()))
        strict = check(docs, _config(max_fallback_title_rate=0.10))
        assert "TITLE_FALLBACK" in _warnings(strict)

    def test_default_used_when_threshold_absent(self):
        assert DEFAULT_MAX_FALLBACK_TITLE_RATE == 0.50  # guards the documented default


# ---------------------------------------------------------------------------
# BODY_EMPTY
# ---------------------------------------------------------------------------


class TestBodyEmpty:
    def test_warns_when_body_expected_and_mostly_empty(self):
        docs = _healthy_batch(5)
        for d in docs:
            d.body = ""
        assert "BODY_EMPTY" in _warnings(check(docs, _config()))

    def test_silent_when_body_not_expected(self):
        """EDGAR sets body=false; empty bodies must not warn."""
        docs = _healthy_batch(5)
        for d in docs:
            d.body = ""
        report = check(docs, _config(expects={"body": False}))
        assert "BODY_EMPTY" not in _warnings(report)

    def test_per_source_threshold_honored(self):
        # 2/5 = 0.40 empty: below the 0.80 default, but a per-source max of 0.30 warns.
        docs = _healthy_batch(5)
        docs[0].body = ""
        docs[1].body = ""
        assert "BODY_EMPTY" not in _warnings(check(docs, _config()))
        assert "BODY_EMPTY" in _warnings(check(docs, _config(max_empty_body_rate=0.30)))


# ---------------------------------------------------------------------------
# Collapse flags (URL / identity / hash) — guarded by MIN_BATCH_FOR_COLLAPSE
# ---------------------------------------------------------------------------


class TestCollapseFlags:
    def test_url_collapse_warns(self):
        docs = _healthy_batch(MIN_BATCH_FOR_COLLAPSE)
        for d in docs:
            d.url = "https://example.com/same"
        assert "URL_COLLAPSE" in _warnings(check(docs, _config()))

    def test_identity_collapse_warns(self):
        docs = _healthy_batch(MIN_BATCH_FOR_COLLAPSE)
        for d in docs:
            d.identity_key = "src::same"
        assert "IDENTITY_COLLAPSE" in _warnings(check(docs, _config()))

    def test_hash_collapse_warns(self):
        docs = _healthy_batch(MIN_BATCH_FOR_COLLAPSE)
        for d in docs:
            d.content_hash = "c" * 64
        assert "HASH_COLLAPSE" in _warnings(check(docs, _config()))

    def test_no_collapse_warning_below_min_batch(self):
        # A tiny batch is trivially "collapsed"; the MIN guard must suppress it.
        docs = _healthy_batch(MIN_BATCH_FOR_COLLAPSE - 1)
        for d in docs:
            d.url = "https://example.com/same"
            d.identity_key = "src::same"
            d.content_hash = "c" * 64
        w = _warnings(check(docs, _config()))
        assert "URL_COLLAPSE" not in w
        assert "IDENTITY_COLLAPSE" not in w
        assert "HASH_COLLAPSE" not in w

    def test_distinct_batch_no_collapse(self):
        assert "COLLAPSE" not in _warnings(check(_healthy_batch(8), _config()))


# ---------------------------------------------------------------------------
# URL_MALFORMED
# ---------------------------------------------------------------------------


class TestUrlMalformed:
    def test_list_repr_url_warns(self):
        docs = _healthy_batch(3)
        docs[0].url = "https://www.sec.gov/Archives/edgar/data/['001-39218']//-index.htm"
        assert "URL_MALFORMED" in _warnings(check(docs, _config()))

    def test_clean_urls_silent(self):
        assert "URL_MALFORMED" not in _warnings(check(_healthy_batch(3), _config()))


# ---------------------------------------------------------------------------
# EMPTY_BATCH — the only flag evaluated on an empty batch
# ---------------------------------------------------------------------------


class TestEmptyBatch:
    def test_warns_when_below_min_records(self):
        report = check([], _config(min_records=1))
        assert "EMPTY_BATCH" in _warnings(report)

    def test_warns_when_thin_batch_below_min(self):
        report = check(_healthy_batch(2), _config(min_records=5))
        assert "EMPTY_BATCH" in _warnings(report)

    def test_silent_when_min_records_absent(self):
        assert "EMPTY_BATCH" not in _warnings(check([], _config()))

    def test_silent_when_batch_meets_min(self):
        report = check(_healthy_batch(5), _config(min_records=5))
        assert "EMPTY_BATCH" not in _warnings(report)


# ---------------------------------------------------------------------------
# Gate never drops / raises (soft contract)
# ---------------------------------------------------------------------------


class TestSoftContract:
    def test_degenerate_batch_warns_but_returns_all_stats(self):
        docs = _healthy_batch(6)
        for d in docs:
            d.url = "https://example.com/same"
            d.body = ""
            d.title = "Unknown"
        report = check(docs, _config())
        # Multiple flags fire, but the report is still well-formed and counts all records.
        assert len(report.warnings) >= 2
        assert report.stats["records"] == 6
