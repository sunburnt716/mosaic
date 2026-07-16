"""
Phase 1 orchestrator — Documents in, Chunks out.

The thin seam that ties document-type dispatch to the chunking strategies: pick the right
chunker for each Document's inferred `document_type` and run it. Chunking itself is pure (no
I/O), and so is this — it operates on in-memory Documents and returns Chunks ready for Phase 2
embedding.

Ordering: Phase 0 inference is expected to have populated `document_type` already; a Document
whose type is still `None` (inference not yet run) falls back to fixed-size chunking rather than
erroring. Reading the store and writing vectors belong to later phases, not here.

  chunk_document(document)   -> list[Chunk]   dispatch one document to its strategy
  chunk_documents(documents) -> list[Chunk]   flatten chunks across many documents
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from extraction.chunkers.registry import get_chunker

if TYPE_CHECKING:
    from extraction.chunk import Chunk
    from ingestion.core.document import Document


def chunk_document(document: "Document", chunked_at: str | None = None) -> "list[Chunk]":
    """Chunk a single Document using the strategy for its inferred document_type."""
    chunker = get_chunker(document.document_type)
    return chunker(document, chunked_at=chunked_at)


def chunk_documents(
    documents: "Iterable[Document]", chunked_at: str | None = None
) -> "list[Chunk]":
    """Chunk many Documents, returning all chunks in document order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunked_at=chunked_at))
    return chunks
