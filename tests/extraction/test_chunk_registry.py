"""
Contract tests for the strategy dispatcher (processing/chunkers/registry.py).

Pins the document_type -> strategy mapping against the inferred type vocabulary, and the
fixed-size fallback for tweets, unknown, and not-yet-inferred (None) documents.
"""

from __future__ import annotations

from extraction.chunkers.fixed import chunk_fixed
from extraction.chunkers.paragraph import chunk_paragraph
from extraction.chunkers.registry import get_chunker
from extraction.chunkers.section import chunk_section
from extraction.type_inference import ARTICLE, FILING, TWEET, UNKNOWN


class TestGetChunker:
    def test_article_maps_to_paragraph(self):
        assert get_chunker(ARTICLE) is chunk_paragraph

    def test_filing_maps_to_section(self):
        assert get_chunker(FILING) is chunk_section

    def test_tweet_and_unknown_fall_back_to_fixed(self):
        assert get_chunker(TWEET) is chunk_fixed
        assert get_chunker(UNKNOWN) is chunk_fixed

    def test_none_falls_back_to_fixed(self):
        assert get_chunker(None) is chunk_fixed
