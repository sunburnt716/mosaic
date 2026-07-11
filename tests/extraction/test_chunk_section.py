"""
Contract tests for the section chunker (processing/chunkers/section.py).

Uses the shared FILING_BODY fixture (real 8-K structure: SEC header, Item markers, Risk Factors,
Forward-Looking Statements) to pin section splitting and the after-header highlight, plus a
synthetic oversized section to pin the fallback.
"""

from __future__ import annotations

from extraction.chunkers.section import chunk_section
from extraction.utils.section_detection import detect_section_headers
from tests.extraction.fixtures import FILING_BODY, make_document


class TestChunkSection:
    def test_one_chunk_per_detected_section(self, fake_tokenizer):
        doc = make_document(body=FILING_BODY, document_type="filing")
        chunks = chunk_section(doc)
        # No preamble (body starts with a header), so chunks == detected sections.
        assert len(chunks) == len(detect_section_headers(FILING_BODY))
        assert chunks[0].text.startswith("UNITED STATES SECURITIES AND EXCHANGE COMMISSION")

    def test_full_spans_are_contiguous_and_cover_the_body(self, fake_tokenizer):
        doc = make_document(body=FILING_BODY, document_type="filing")
        chunks = chunk_section(doc)
        assert chunks[0].full_span[0] == 0
        for prev, nxt in zip(chunks, chunks[1:]):
            assert prev.full_span[1] == nxt.full_span[0]
        assert chunks[-1].full_span[1] == len(FILING_BODY)

    def test_highlight_after_header(self, fake_tokenizer):
        body = "RISK FACTORS\nThe company faces risks. Markets are volatile.\n"
        doc = make_document(body=body, document_type="filing")
        chunk = chunk_section(doc)[0]
        assert doc.body[chunk.highlight_span[0] : chunk.highlight_span[1]] == (
            "The company faces risks."
        )

    def test_section_label_is_header_text(self, fake_tokenizer):
        body = "RISK FACTORS\nThe company faces risks. Markets are volatile.\n"
        doc = make_document(body=body, document_type="filing")
        chunk = chunk_section(doc)[0]
        assert chunk.section_label == "RISK FACTORS"
        assert chunk.ordinal == 0

    def test_no_headers_is_single_section(self, fake_tokenizer):
        body = "Just some plain prose without any section headers to be found here.\n"
        chunks = chunk_section(make_document(body=body, document_type="filing"))
        assert len(chunks) == 1
        assert chunks[0].text == body

    def test_preamble_before_first_header(self, fake_tokenizer):
        body = "Intro prose before any header line at all.\nRISK FACTORS\nRisk body here.\n"
        chunks = chunk_section(make_document(body=body, document_type="filing"))
        assert len(chunks) == 2
        assert chunks[0].text.startswith("Intro prose")
        assert chunks[1].text.startswith("RISK FACTORS")
        assert chunks[0].section_label is None
        assert chunks[1].section_label == "RISK FACTORS"

    def test_oversized_section_falls_back_to_fixed(self, fake_tokenizer):
        body = "OVERVIEW\n" + " ".join(f"t{i}" for i in range(600)) + "\n"
        doc = make_document(id="f", body=body, document_type="filing")
        chunks = chunk_section(doc, max_section_tokens=100, fallback_strategy="fixed")
        assert len(chunks) > 1
        assert [c.chunk_id for c in chunks] == [f"f#{i}" for i in range(len(chunks))]
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        assert chunks[0].text.startswith("OVERVIEW")
        # Sub-chunks from the fallback split are still part of the OVERVIEW section.
        assert all(c.section_label == "OVERVIEW" for c in chunks)

    def test_empty_body_yields_no_chunks(self, fake_tokenizer):
        assert chunk_section(make_document(body="", document_type="filing")) == []
