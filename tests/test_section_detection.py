"""Contract tests for extraction/utils/section_detection.py.

Pins the regex heuristics that decide what a section header looks like: all-caps titles,
numbered items, and short keyword headings — while ordinary prose is left alone. Returned
spans must locate the header line in the source text (the section chunker splits on them).
"""

from extraction.utils.section_detection import detect_section_headers


class TestDetectSectionHeaders:
    def test_all_caps_line_is_a_header(self):
        text = "RISK FACTORS\nThe company faces many risks.\n"
        headers = detect_section_headers(text)
        assert [text[s:e] for s, e in headers] == ["RISK FACTORS"]

    def test_numbered_item_is_a_header(self):
        text = "Item 1A. Risk Factors\nSome body text here about risk.\n"
        headers = detect_section_headers(text)
        assert [text[s:e] for s, e in headers] == ["Item 1A. Risk Factors"]

    def test_keyword_heading_is_detected(self):
        text = "Legal Proceedings\nThe company is party to litigation.\n"
        headers = detect_section_headers(text)
        assert [text[s:e] for s, e in headers] == ["Legal Proceedings"]

    def test_prose_is_not_a_header(self):
        text = (
            "The management team believes financial performance will improve as the "
            "business grows and legal proceedings conclude over the coming year ahead.\n"
        )
        assert detect_section_headers(text) == []

    def test_multiple_headers_in_document_order(self):
        text = "RISK FACTORS\nbody\nItem 2. Properties\nmore body\n"
        headers = detect_section_headers(text)
        assert [text[s:e] for s, e in headers] == ["RISK FACTORS", "Item 2. Properties"]

    def test_no_headers_returns_empty(self):
        assert detect_section_headers("just a plain sentence with no headers.") == []

    def test_span_excludes_leading_indentation(self):
        text = "  RISK FACTORS  \nbody\n"
        ((start, end),) = detect_section_headers(text)
        assert text[start:end] == "RISK FACTORS"
