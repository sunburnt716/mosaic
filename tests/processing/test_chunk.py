"""
Contract tests for the Chunk dataclass and its builders (processing/chunk.py).

Pins the output contract: deterministic chunk_id, provenance copied verbatim off the parent
Document (must reach Chroma for citation), and correct absolute offsets + text slicing from
`materialize_chunks` (including the base/start_ordinal shift the section chunker relies on).
"""

from __future__ import annotations

from datetime import datetime

from processing.chunk import build_chunk, materialize_chunks
from tests.processing.fixtures import make_document


class TestBuildChunk:
    def test_chunk_id_is_document_id_plus_ordinal(self):
        doc = make_document(id="deadbeef")
        chunk = build_chunk(doc, 3, "text", (0, 4), (0, 4), chunked_at="t")
        assert chunk.chunk_id == "deadbeef#3"
        assert chunk.document_id == "deadbeef"
        assert chunk.ordinal == 3

    def test_section_label_defaults_to_none(self):
        chunk = build_chunk(make_document(), 0, "text", (0, 4), (0, 4), chunked_at="t")
        assert chunk.section_label is None

    def test_section_label_passed_through(self):
        chunk = build_chunk(
            make_document(), 0, "text", (0, 4), (0, 4), chunked_at="t", section_label="RISK FACTORS"
        )
        assert chunk.section_label == "RISK FACTORS"

    def test_provenance_copied_from_document(self):
        published = datetime(2026, 6, 30, 12, 0)
        doc = make_document(
            title="Big News",
            url="https://example.com/a",
            source_name="Reuters",
            tier=1,
            published_date=published,
        )
        chunk = build_chunk(doc, 0, "text", (0, 4), (0, 4), chunked_at="t")
        assert chunk.title == "Big News"
        assert chunk.url == "https://example.com/a"
        assert chunk.source_name == "Reuters"
        assert chunk.tier == 1
        assert chunk.published_date == published

    def test_chunked_at_injectable(self):
        chunk = build_chunk(make_document(), 0, "t", (0, 1), (0, 1), chunked_at="STAMP")
        assert chunk.chunked_at == "STAMP"

    def test_chunked_at_defaults_to_iso_utc(self):
        chunk = build_chunk(make_document(), 0, "t", (0, 1), (0, 1))
        parsed = datetime.fromisoformat(chunk.chunked_at)
        assert parsed.tzinfo is not None


class TestMaterializeChunks:
    def test_text_sliced_from_body_matches_full_span(self):
        doc = make_document(body="0123456789")
        plans = [((0, 4), (0, 2)), ((4, 10), (4, 6))]
        chunks = materialize_chunks(doc, plans, chunked_at="t")
        assert [c.text for c in chunks] == ["0123", "456789"]
        assert chunks[0].full_span == (0, 4)
        assert chunks[0].highlight_span == (0, 2)

    def test_base_offset_shifts_spans_and_ordinals(self):
        doc = make_document(body="0123456789")
        chunks = materialize_chunks(
            doc, [((0, 3), (0, 1))], base=4, start_ordinal=2, chunked_at="t"
        )
        assert chunks[0].full_span == (4, 7)
        assert chunks[0].highlight_span == (4, 5)
        assert chunks[0].text == "456"
        assert chunks[0].chunk_id.endswith("#2")
        assert chunks[0].ordinal == 2

    def test_empty_plans_yield_no_chunks(self):
        assert materialize_chunks(make_document(), [], chunked_at="t") == []

    def test_section_label_applies_to_every_chunk_in_call(self):
        doc = make_document(body="0123456789")
        plans = [((0, 4), (0, 2)), ((4, 10), (4, 6))]
        chunks = materialize_chunks(doc, plans, chunked_at="t", section_label="RISK FACTORS")
        assert [c.section_label for c in chunks] == ["RISK FACTORS", "RISK FACTORS"]

    def test_section_label_defaults_to_none(self):
        doc = make_document(body="0123456789")
        chunks = materialize_chunks(doc, [((0, 4), (0, 2))], chunked_at="t")
        assert chunks[0].section_label is None
