"""Contract tests for the strategy dispatcher (extraction/chunkers/registry.py).

Pins the doc_type → strategy mapping against the values the pipeline actually produces
("article", "filing") and the fixed-size fallback for anything unmapped.
"""

from extraction.chunkers.fixed import chunk_fixed
from extraction.chunkers.paragraph import chunk_paragraph
from extraction.chunkers.registry import get_chunker
from extraction.chunkers.section import chunk_section


class TestGetChunker:
    def test_article_maps_to_paragraph(self):
        assert get_chunker("article") is chunk_paragraph

    def test_filing_maps_to_section(self):
        assert get_chunker("filing") is chunk_section

    def test_unknown_type_falls_back_to_fixed(self):
        assert get_chunker("blog_post") is chunk_fixed
        assert get_chunker("") is chunk_fixed
