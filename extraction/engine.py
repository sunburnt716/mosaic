"""Extraction orchestrator — Documents in, Chunks out (Phase 1 -> Phase 2 handoff).

The thin seam that ties document-type dispatch to the chunking strategies: pick the right
chunker for each Document's `doc_type` and run it. Chunking itself is pure (no I/O), and so
is this — it operates on in-memory Documents and returns Chunks ready to be embedded and
written to Chroma by the (future) Phase 2 embedding stage.

Reading `status: unprocessed` Documents from the raw store and advancing their status after
processing is deliberately *not* here yet: that store coupling belongs with the embedding
stage that persists the results (CLAUDE.md's "extraction reads on its own clock"). Phase 1
stops at producing Chunks.

  chunk_document(document)   -> list[Chunk]   dispatch one document to its strategy
  chunk_documents(documents) -> list[Chunk]   flatten chunks across many documents
"""

from typing import TYPE_CHECKING, Iterable

from extraction.chunkers.registry import get_chunker

if TYPE_CHECKING:
    from extraction.chunk import Chunk
    from ingestion.core.document import Document


def chunk_document(document: "Document", chunked_at: str | None = None) -> "list[Chunk]":
    """Chunk a single Document using the strategy for its doc_type."""
    chunker = get_chunker(document.doc_type)
    return chunker(document, chunked_at=chunked_at)


def chunk_documents(
    documents: "Iterable[Document]", chunked_at: str | None = None
) -> "list[Chunk]":
    """Chunk many Documents, returning all chunks in document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunked_at=chunked_at))
    return chunks
