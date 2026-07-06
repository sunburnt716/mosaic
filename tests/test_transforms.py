"""Tests for ingestion/pipeline/transforms.py.

All tests run offline — no network, no live APIs. The edgar_filing_url transform
is tested against the title patterns observed in tests/fixtures/sec-edgar.xml.
"""

import pytest

from ingestion.pipeline.transforms import edgar_filing_url, get_transform

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_known_transform(self):
        fn = get_transform("edgar_filing_url")
        assert callable(fn)

    def test_get_unknown_transform_raises(self):
        with pytest.raises(ValueError, match="Unknown transform"):
            get_transform("does_not_exist")


# ---------------------------------------------------------------------------
# edgar_filing_url
# ---------------------------------------------------------------------------


class TestEdgarFilingUrl:
    """Tests derived from the title patterns observed in tests/fixtures/sec-edgar.xml."""

    def _run(self, raw: dict) -> dict:
        return edgar_filing_url(raw, config=None)

    def test_cleans_standard_title(self):
        raw = {
            "url": "https://www.sec.gov/Archives/edgar/data/1649989/000110465926078337/0001104659-26-078337-index.htm",
            "title": "8-K - Outlook Therapeutics, Inc. (0001649989) (Filer)",
            "raw_body": "<b>Filed:</b> 2026-06-26",
            "published": "2026-06-26T17:30:25-04:00",
            "source_article_id": "urn:tag:sec.gov,2008:accession-number=0001104659-26-078337",
        }
        result = self._run(raw)
        assert result["title"] == "8-K — Outlook Therapeutics, Inc."
        assert result["form"] == "8-K"

    def test_extracts_form_type(self):
        raw = {
            "url": "https://www.sec.gov/Archives/edgar/data/61004/000143774926021864/0001437749-26-021864-index.htm",
            "title": "10-K - LGL GROUP INC (0000061004) (Filer)",
            "raw_body": "",
            "published": "2026-06-26T17:30:21-04:00",
            "source_article_id": "urn:tag:sec.gov,2008:accession-number=0001437749-26-021864",
        }
        result = self._run(raw)
        assert result["form"] == "10-K"
        assert result["title"] == "10-K — LGL GROUP INC"

    def test_url_unchanged(self):
        """The transform must not alter the URL — it already is canonical from the feed."""
        original_url = "https://www.sec.gov/Archives/edgar/data/1649989/000110465926078337/0001104659-26-078337-index.htm"
        raw = {
            "url": original_url,
            "title": "8-K - Outlook Therapeutics, Inc. (0001649989) (Filer)",
            "raw_body": "",
            "published": "2026-06-26T17:30:25-04:00",
            "source_article_id": "acc-001",
        }
        result = self._run(raw)
        assert result["url"] == original_url

    def test_url_contains_no_list_repr_or_double_slash(self):
        """Hard guarantee from the spec: the URL must never have [ ] ' or //."""
        raw = {
            "url": "https://www.sec.gov/Archives/edgar/data/1649989/000110465926078337/0001104659-26-078337-index.htm",
            "title": "8-K - Some Corp (0001649989) (Filer)",
            "raw_body": "",
            "published": "2026-06-26T17:30:25-04:00",
            "source_article_id": "acc-002",
        }
        result = self._run(raw)
        assert "[" not in result["url"]
        assert "]" not in result["url"]
        assert "'" not in result["url"]
        # Allow https:// at the start; disallow // elsewhere
        assert "//" not in result["url"].replace("https://", "")

    def test_unrecognised_title_passes_through(self):
        """If the title doesn't match the expected pattern, leave it as-is and set form=''."""
        raw = {
            "url": "https://www.sec.gov/Archives/edgar/data/123/000123/0001-index.htm",
            "title": "Some unexpected title format",
            "raw_body": "",
            "published": "2026-06-26T17:30:25-04:00",
            "source_article_id": "acc-003",
        }
        result = self._run(raw)
        assert result["title"] == "Some unexpected title format"
        assert result["form"] == ""

    def test_does_not_mutate_input(self):
        """Transform must return a new dict; caller's dict must be unchanged."""
        raw = {
            "url": "https://www.sec.gov/Archives/edgar/data/123/000123/0001-index.htm",
            "title": "8-K - Corp X (0000001) (Filer)",
            "raw_body": "",
            "published": "2026-06-26T17:30:25-04:00",
            "source_article_id": "acc-004",
        }
        original_title = raw["title"]
        self._run(raw)
        assert raw["title"] == original_title  # untouched

    def test_multiword_entity_name(self):
        """Entity names with commas or mixed punctuation should round-trip cleanly."""
        raw = {
            "url": "https://www.sec.gov/Archives/edgar/data/1/0001-index.htm",
            "title": "8-K - Acme Holdings, Ltd. (0000001) (Filer)",
            "raw_body": "",
            "published": "2026-06-26T00:00:00-04:00",
            "source_article_id": "acc-005",
        }
        result = self._run(raw)
        assert result["title"] == "8-K — Acme Holdings, Ltd."


# ---------------------------------------------------------------------------
# Integration: fixture -> feedparser -> transform -> normalize
# ---------------------------------------------------------------------------


class TestEdgarFixtureRoundtrip:
    """Run the real captured fixture through the full parse→transform→normalize chain."""

    @pytest.mark.parametrize("entry_index", [0, 1, 2])
    def test_normalize_produces_valid_document(self, entry_index, fetched_at):
        from pathlib import Path

        import feedparser

        from ingestion.pipeline.normalizer import normalize
        from tests.conftest import make_source_config

        fixture = (Path(__file__).parent / "fixtures" / "sec-edgar.xml").read_bytes()
        parsed = feedparser.parse(fixture)

        if entry_index >= len(parsed.entries):
            pytest.skip(f"Fixture has fewer than {entry_index + 1} entries")

        entry = parsed.entries[entry_index]
        raw = {
            "url": entry.get("link"),
            "title": entry.get("title"),
            "raw_body": entry.get("summary", ""),
            "published": entry.get("published") or entry.get("updated"),
            "source_article_id": entry.get("id") or entry.get("link"),
            "raw_payload": dict(entry),
        }

        config = make_source_config(
            name="sec-edgar",
            adapter="rss",
            tier=0,
            url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom",
            doc_type="filing",
            transform="edgar_filing_url",
            expects={"title": True, "url": True, "body": False},
        )

        doc = normalize(raw, config, fetched_at)

        # Title must not be the fallback and must contain "—"
        assert "Unknown" not in doc.title
        assert "—" in doc.title

        # URL must be a proper EDGAR archive URL
        assert doc.url.startswith("https://www.sec.gov/Archives/edgar/data/")
        assert "[" not in doc.url
        assert "]" not in doc.url
        assert "'" not in doc.url
        assert "//" not in doc.url.replace("https://", "")

        # Tier stamped from config
        assert doc.tier == 0
        assert doc.doc_type == "filing"

        # Date must parse
        assert doc.published_date is not None

        # Body may be empty at discovery — but the HTML summary gives us a description
        # In practice it will be non-empty since summary has filing metadata
        assert isinstance(doc.body, str)

    def test_first_entry_exact_archive_url_and_title(self, fetched_at):
        """Pin the exact normalized URL + title for a known fixture entry (regression).

        The getcurrent feed already carries the canonical archive URL (CIK + accession,
        dashes stripped) in the entry link; the transform must pass it through untouched
        and clean only the title.
        """
        from pathlib import Path

        import feedparser

        from ingestion.pipeline.normalizer import normalize
        from tests.conftest import make_source_config

        fixture = (Path(__file__).parent / "fixtures" / "sec-edgar.xml").read_bytes()
        entry = feedparser.parse(fixture).entries[0]
        raw = {
            "url": entry.get("link"),
            "title": entry.get("title"),
            "raw_body": entry.get("summary", ""),
            "published": entry.get("published") or entry.get("updated"),
            "source_article_id": entry.get("id") or entry.get("link"),
            "raw_payload": dict(entry),
        }
        config = make_source_config(
            name="sec-edgar",
            adapter="rss",
            tier=0,
            url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom",
            doc_type="filing",
            transform="edgar_filing_url",
        )
        doc = normalize(raw, config, fetched_at)

        assert doc.url == (
            "https://www.sec.gov/Archives/edgar/data/1649989/"
            "000110465926078337/0001104659-26-078337-index.htm"
        )
        assert doc.title == "8-K — Outlook Therapeutics, Inc."
