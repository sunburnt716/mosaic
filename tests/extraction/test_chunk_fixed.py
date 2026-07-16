"""
Contract tests for the fixed-size chunker (processing/chunkers/fixed.py).

Uses the word-level FakeTokenizer, so "tokens" here are whitespace words: a body of N words with
chunk_size/overlap in words lets us pin window boundaries, overlap, ordinals, and the fixed-chunk
rule that highlight_span == full_span.
"""

from __future__ import annotations

import pytest

from extraction.chunkers.fixed import chunk_fixed
from tests.extraction.fixtures import make_document

# Ten tokens "w0".."w9", each 2 chars, single-space separated (token i at [i*3, i*3+2]).
_BODY = " ".join(f"w{i}" for i in range(10))


class TestChunkFixed:
    def test_windows_with_overlap(self, fake_tokenizer):
        chunks = chunk_fixed(make_document(body=_BODY), chunk_size=4, overlap=1, chunked_at="t")
        assert [c.text for c in chunks] == [
            "w0 w1 w2 w3",
            "w3 w4 w5 w6",
            "w6 w7 w8 w9",
        ]

    def test_adjacent_chunks_share_overlap_token(self, fake_tokenizer):
        chunks = chunk_fixed(make_document(body=_BODY), chunk_size=4, overlap=1, chunked_at="t")
        assert chunks[0].text.endswith("w3")
        assert chunks[1].text.startswith("w3")

    def test_highlight_equals_full_span(self, fake_tokenizer):
        chunks = chunk_fixed(make_document(body=_BODY), chunk_size=4, overlap=1, chunked_at="t")
        assert all(c.full_span == c.highlight_span for c in chunks)

    def test_ordinals_are_contiguous(self, fake_tokenizer):
        chunks = chunk_fixed(
            make_document(id="d", body=_BODY), chunk_size=4, overlap=1, chunked_at="t"
        )
        assert [c.chunk_id for c in chunks] == ["d#0", "d#1", "d#2"]
        assert [c.ordinal for c in chunks] == [0, 1, 2]

    def test_section_label_is_none(self, fake_tokenizer):
        chunks = chunk_fixed(make_document(body=_BODY), chunk_size=4, overlap=1, chunked_at="t")
        assert all(c.section_label is None for c in chunks)

    def test_body_smaller_than_window_is_one_chunk(self, fake_tokenizer):
        chunks = chunk_fixed(make_document(body="alpha beta gamma"), chunked_at="t")
        assert len(chunks) == 1
        assert chunks[0].text == "alpha beta gamma"

    def test_empty_body_yields_no_chunks(self, fake_tokenizer):
        assert chunk_fixed(make_document(body=""), chunked_at="t") == []

    def test_overlap_not_smaller_than_chunk_size_raises(self, fake_tokenizer):
        with pytest.raises(ValueError):
            chunk_fixed(make_document(body=_BODY), chunk_size=4, overlap=4)
