"""
Contract tests for the paragraph chunker (processing/chunkers/paragraph.py).

Paragraphs are the blank-line blocks that text_metrics.paragraph_spans defines. With the
word-level FakeTokenizer, min_paragraph_tokens is a word count, letting us pin the orphan-merge
behaviour (forward, and tail-folded-back) and the first-sentence highlight.
"""

from __future__ import annotations

from processing.chunkers.paragraph import chunk_paragraph
from tests.processing.fixtures import make_document


class TestChunkParagraph:
    def test_one_chunk_per_paragraph_when_all_large(self, fake_tokenizer):
        body = "para one has words.\n\npara two has words.\n\npara three here now."
        chunks = chunk_paragraph(make_document(body=body), min_paragraph_tokens=1)
        assert [c.text for c in chunks] == [
            "para one has words.",
            "para two has words.",
            "para three here now.",
        ]

    def test_small_middle_paragraph_merges_forward(self, fake_tokenizer):
        body = "aaa bbb ccc ddd eee\n\nfff\n\nggg hhh iii jjj kkk"
        chunks = chunk_paragraph(make_document(body=body), min_paragraph_tokens=3)
        assert [c.text for c in chunks] == [
            "aaa bbb ccc ddd eee",
            "fff\n\nggg hhh iii jjj kkk",
        ]

    def test_small_tail_paragraph_folds_back(self, fake_tokenizer):
        body = "aaa bbb ccc ddd eee\n\nfff"
        chunks = chunk_paragraph(make_document(body=body), min_paragraph_tokens=3)
        assert [c.text for c in chunks] == ["aaa bbb ccc ddd eee\n\nfff"]

    def test_highlight_is_first_sentence(self, fake_tokenizer):
        body = "First sentence. Second one here.\n\nAnother paragraph entirely."
        doc = make_document(body=body)
        first = chunk_paragraph(doc, min_paragraph_tokens=1)[0]
        assert doc.body[first.highlight_span[0] : first.highlight_span[1]] == "First sentence."

    def test_full_span_slices_back_to_text(self, fake_tokenizer):
        body = "para one has words.\n\npara two has words."
        doc = make_document(body=body)
        for chunk in chunk_paragraph(doc, min_paragraph_tokens=1):
            assert doc.body[chunk.full_span[0] : chunk.full_span[1]] == chunk.text

    def test_provenance_carried(self, fake_tokenizer):
        doc = make_document(source_name="Reuters", tier=1, url="https://x.test/a")
        chunk = chunk_paragraph(doc, min_paragraph_tokens=1)[0]
        assert chunk.source_name == "Reuters"
        assert chunk.tier == 1
        assert chunk.url == "https://x.test/a"

    def test_empty_body_yields_no_chunks(self, fake_tokenizer):
        assert chunk_paragraph(make_document(body="")) == []

    def test_section_label_is_none(self, fake_tokenizer):
        body = "para one has words.\n\npara two has words."
        chunks = chunk_paragraph(make_document(body=body), min_paragraph_tokens=1)
        assert all(c.section_label is None for c in chunks)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
