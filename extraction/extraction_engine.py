"""
Phase 2+3 orchestrator — Documents in, vectors in Chroma out.

`extract()` is a pure function: it takes in-memory Documents, runs them through the
full Phase 0→1→2→3 chain, and writes the resulting vectors to Chroma.  It has no global
state and performs no I/O beyond the injected `embedder` and `chroma_store` dependencies,
making it equally usable from the hot-path ingestion loop and the cold-path query-time
fallback without modification.

Per-document isolation: an error processing one Document is caught, logged, and counted
in `ExtractionResult.errors`; the remaining Documents continue.  An empty chunk list
(e.g. an extremely short document that produces no chunks) is silently skipped — it is
not counted as an error.

  extract(documents, embedder, chroma_store, ...)  -> ExtractionResult
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from extraction.engine import chunk_document
from extraction.type_inference import infer_document_type
from extraction.validation import validate_document

if TYPE_CHECKING:
    from extraction.chroma_store import ChromaVectorStore
    from extraction.chunk import Chunk
    from extraction.embedder import Embedder
    from ingestion.core.document import Document

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Summary of one `extract()` run."""

    documents_processed: int = 0
    chunks_written: int = 0
    errors: list[str] = field(default_factory=list)


def extract(
    documents: Iterable["Document"],
    embedder: "Embedder",
    chroma_store: "ChromaVectorStore",
    *,
    source_hints: dict[str, str] | None = None,
) -> ExtractionResult:
    """Run the Phase 0→1→2→3 pipeline for each Document and write vectors to Chroma.

    `source_hints` is an optional mapping from `source_name` to a document-type hint
    (e.g. `{"sec-edgar": "filing"}`).  It is forwarded to `infer_document_type` as an
    advisory; the heuristic may still override it.
    """
    result = ExtractionResult()

    for doc in documents:
        try:
            _process_one(doc, embedder, chroma_store, source_hints, result)
        except Exception as exc:  # noqa: BLE001
            msg = f"[{doc.source_name}/{doc.id}] extraction failed: {exc}"
            logger.error(msg)
            result.errors.append(msg)

    return result


def _process_one(
    doc: "Document",
    embedder: "Embedder",
    chroma_store: "ChromaVectorStore",
    source_hints: dict[str, str] | None,
    result: ExtractionResult,
) -> None:
    """Run the pipeline for a single Document, updating `result` in place."""
    # Phase 0: infer document type and attach it to an updated (frozen) copy.
    hint = (source_hints or {}).get(doc.source_name)
    inferred_type = infer_document_type(doc, source_hints={doc.source_name: hint} if hint else None)
    typed_doc = dataclasses.replace(doc, document_type=inferred_type)

    # Phase 0 (cont.): validate structure vs inferred type; warnings are logged but
    # never block processing — even degenerate docs flow through.
    validation = validate_document(typed_doc, inferred_type)
    if validation.warnings:
        for w in validation.warnings:
            logger.warning("[%s/%s] %s", doc.source_name, doc.id, w)

    # Phase 1: chunk by document type.
    chunks: list[Chunk] = chunk_document(typed_doc)
    if not chunks:
        return

    # Phase 2: embed the full-span text of each chunk.
    vectors = embedder.embed([c.text for c in chunks])

    # Phase 3: write vectors + citation metadata to Chroma.
    chroma_store.upsert(chunks, vectors)

    result.documents_processed += 1
    result.chunks_written += len(chunks)
