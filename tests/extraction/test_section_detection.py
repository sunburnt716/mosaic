"""
Contract tests for processing/utils/section_detection.py.

Pins that header detection finds filing markers (reused from text_metrics) and all-caps titles
while leaving prose alone. Uses the shared FILING_BODY / ARTICLE_BODY fixtures so detection is
exercised against realistic filing structure.
"""

from __future__ import annotations

from extraction.utils.section_detection import detect_section_headers
from tests.extraction.fixtures import ARTICLE_BODY, FILING_BODY


class TestDetectSectionHeaders:
    def test_finds_all_caps_and_marker_headers_in_a_filing(self):
        headers = [FILING_BODY[s:e] for s, e in detect_section_headers(FILING_BODY)]
        assert "UNITED STATES SECURITIES AND EXCHANGE COMMISSION" in headers
        assert "FORM 8-K" in headers
        assert "Item 1.01 Entry into a Material Definitive Agreement." in headers
        assert "Risk Factors" in headers
        assert "Forward-Looking Statements" in headers

    def test_headers_are_in_document_order(self):
        spans = detect_section_headers(FILING_BODY)
        assert spans == sorted(spans)

    def test_article_prose_has_no_headers(self):
        assert detect_section_headers(ARTICLE_BODY) == []

    def test_all_caps_line(self):
        text = "RISK FACTORS\nThe company faces many risks.\n"
        assert [text[s:e] for s, e in detect_section_headers(text)] == ["RISK FACTORS"]

    def test_numbered_item(self):
        text = "Item 1A. Risk Factors\nSome body text about risk here.\n"
        assert [text[s:e] for s, e in detect_section_headers(text)] == ["Item 1A. Risk Factors"]

    def test_span_excludes_indentation(self):
        text = "  RISK FACTORS  \nbody\n"
        ((start, end),) = detect_section_headers(text)
        assert text[start:end] == "RISK FACTORS"
