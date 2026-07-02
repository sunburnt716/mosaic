"""Contract tests for the section chunker (extraction/chunkers/section.py).

Uses a synthetic filing body with known headers (real EDGAR discovery is metadata-only,
so its Documents have empty bodies — see the module docstring). Pins section splitting,
the header-less preamble, the after-header highlight, and the oversized-section fallback.
"""

from extraction.chunkers.section import chunk_section
from tests.conftest import make_document

_FILING = (
    "RISK FACTORS\n"
    "The company faces risks. Markets are volatile.\n"
    "Item 2. Properties\n"
    "We own offices in several cities.\n"
)


class TestChunkSection:
    def test_splits_at_headers(self, fake_tokenizer):
        chunks = chunk_section(make_document(body=_FILING, doc_type="filing"))
        assert [c.text for c in chunks] == [
            "RISK FACTORS\nThe company faces risks. Markets are volatile.\n",
            "Item 2. Properties\nWe own offices in several cities.\n",
        ]

    def test_highlight_is_first_sentence_after_header(self, fake_tokenizer):
        doc = make_document(body=_FILING, doc_type="filing")
        chunks = chunk_section(doc)
        assert doc.body[chunks[0].highlight_span[0] : chunks[0].highlight_span[1]] == (
            "The company faces risks."
        )
        assert doc.body[chunks[1].highlight_span[0] : chunks[1].highlight_span[1]] == (
            "We own offices in several cities."
        )

    def test_no_headers_is_single_section(self, fake_tokenizer):
        body = "Just some plain text without any section headers here.\n"
        doc = make_document(body=body, doc_type="filing")
        chunks = chunk_section(doc)
        assert len(chunks) == 1
        assert chunks[0].text == body

    def test_preamble_before_first_header(self, fake_tokenizer):
        body = "Intro text before any header line here.\nRISK FACTORS\nRisk body.\n"
        chunks = chunk_section(make_document(body=body, doc_type="filing"))
        assert len(chunks) == 2
        assert chunks[0].text.startswith("Intro text")
        assert chunks[1].text.startswith("RISK FACTORS")

    def test_oversized_section_falls_back_to_fixed(self, fake_tokenizer):
        body = "OVERVIEW\n" + " ".join(f"t{i}" for i in range(600)) + "\n"
        doc = make_document(id="filingid", body=body, doc_type="filing")
        chunks = chunk_section(doc, max_section_tokens=100, fallback_strategy="fixed")
        assert len(chunks) > 1
        assert [c.chunk_id for c in chunks] == [f"filingid#{i}" for i in range(len(chunks))]
        assert chunks[0].text.startswith("OVERVIEW")

    def test_empty_body_yields_no_chunks(self, fake_tokenizer):
        assert chunk_section(make_document(body="", doc_type="filing")) == []
