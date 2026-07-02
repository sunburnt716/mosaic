"""Contract tests for the extraction orchestrator (extraction/engine.py).

Pins that `chunk_document` dispatches by `doc_type` and that `chunk_documents` flattens
across many documents in order — the Phase 1 → Phase 2 handoff seam.
"""

from extraction.engine import chunk_document, chunk_documents
from tests.conftest import make_document


class TestChunkDocument:
    def test_article_dispatches_to_paragraph(self, fake_tokenizer):
        doc = make_document(
            doc_type="article", body="First sentence here. Second sentence follows."
        )
        chunks = chunk_document(doc)
        # Paragraph strategy highlights the first sentence; fixed would highlight the
        # whole chunk (highlight_span == full_span). This proves the dispatch wiring.
        assert chunks[0].highlight_span != chunks[0].full_span
        assert doc.body[chunks[0].highlight_span[0] : chunks[0].highlight_span[1]] == (
            "First sentence here."
        )

    def test_filing_dispatches_to_section(self, fake_tokenizer):
        body = "RISK FACTORS\nThe risk body sentence here.\n"
        chunks = chunk_document(make_document(doc_type="filing", body=body))
        assert len(chunks) == 1
        assert chunks[0].text.startswith("RISK FACTORS")

    def test_unknown_type_uses_fixed_fallback(self, fake_tokenizer):
        chunks = chunk_document(make_document(doc_type="mystery", body="a b c d e"))
        assert len(chunks) == 1
        assert chunks[0].full_span == chunks[0].highlight_span  # fixed-chunk signature

    def test_chunked_at_propagates(self, fake_tokenizer):
        chunks = chunk_document(
            make_document(doc_type="article", body="one two three."),
            chunked_at="STAMP",
        )
        assert all(c.chunked_at == "STAMP" for c in chunks)


class TestChunkDocuments:
    def test_flattens_in_document_order(self, fake_tokenizer):
        docs = [
            make_document(id="d0", doc_type="article", body="alpha beta gamma."),
            make_document(id="d1", doc_type="article", body="delta epsilon zeta."),
        ]
        chunks = chunk_documents(docs)
        assert [c.document_id for c in chunks] == ["d0", "d1"]

    def test_empty_iterable_yields_no_chunks(self, fake_tokenizer):
        assert chunk_documents([]) == []
