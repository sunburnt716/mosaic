"""
Contract tests for the Phase 1 orchestrator (processing/engine.py).

Pins that chunk_document dispatches by the inferred document_type and that chunk_documents
flattens across many documents in order — the Phase 1 -> Phase 2 handoff seam.
"""

from __future__ import annotations

from processing.engine import chunk_document, chunk_documents
from tests.processing.fixtures import make_document


class TestChunkDocument:
    def test_article_dispatches_to_paragraph(self, fake_tokenizer):
        doc = make_document(
            document_type="article", body="First sentence here. Second sentence follows."
        )
        chunk = chunk_document(doc)[0]
        # Paragraph strategy highlights the first sentence; fixed would highlight the whole
        # chunk (highlight_span == full_span). This proves the dispatch wiring.
        assert chunk.highlight_span != chunk.full_span
        assert doc.body[chunk.highlight_span[0] : chunk.highlight_span[1]] == "First sentence here."

    def test_filing_dispatches_to_section(self, fake_tokenizer):
        doc = make_document(document_type="filing", body="RISK FACTORS\nThe risk body sentence.\n")
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text.startswith("RISK FACTORS")

    def test_tweet_uses_fixed_fallback(self, fake_tokenizer):
        doc = make_document(document_type="tweet", body="$AAPL up on guidance #stocks")
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].full_span == chunks[0].highlight_span  # fixed-chunk signature

    def test_none_type_uses_fixed_fallback(self, fake_tokenizer):
        doc = make_document(body="a b c d e")  # document_type defaults to None
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].full_span == chunks[0].highlight_span

    def test_chunked_at_propagates(self, fake_tokenizer):
        doc = make_document(document_type="article", body="one two three.")
        chunks = chunk_document(doc, chunked_at="STAMP")
        assert all(c.chunked_at == "STAMP" for c in chunks)


class TestChunkDocuments:
    def test_flattens_in_document_order(self, fake_tokenizer):
        docs = [
            make_document(id="d0", document_type="article", body="alpha beta gamma."),
            make_document(id="d1", document_type="article", body="delta epsilon zeta."),
        ]
        assert [c.document_id for c in chunk_documents(docs)] == ["d0", "d1"]

    def test_empty_iterable_yields_no_chunks(self, fake_tokenizer):
        assert chunk_documents([]) == []
