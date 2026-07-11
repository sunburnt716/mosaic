"""
Cold-path extraction — process a single Document on demand, if it isn't already.

This is the callable side of the cold path (see CLAUDE.md "Extraction layer"): a
document from a `processing_mode: cold` source sits in the raw store as `unprocessed`
until something asks for it. `ensure_processed()` is that ask — a query-time cache-miss
on Chroma (retrieval finds no vectors for a doc_id it wants to cite) would call this
before ranking.

No caller is wired in yet. The `retrieval/` layer that would call this on a cache-miss
doesn't exist in this codebase yet; this function is built and tested so retrieval can
call it once that layer exists (CLAUDE.md Phase map, Phase 5).

  ensure_processed(doc_id, raw_store, embedder, chroma_store) -> bool
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from extraction.extraction_engine import extract

if TYPE_CHECKING:
    from extraction.chroma_store import ChromaVectorStore
    from extraction.embedder import Embedder
    from ingestion.storage.raw_store import RawStore


def ensure_processed(
    doc_id: str,
    raw_store: "RawStore",
    embedder: "Embedder",
    chroma_store: "ChromaVectorStore",
    *,
    source_hints: dict[str, str] | None = None,
) -> bool:
    """Extract and embed the document for `doc_id` if it isn't already processed.

    Returns True if the document ends up processed (either it already was, or this
    call successfully processed it); False if the document doesn't exist in the raw
    store, or extraction failed for it.
    """
    doc = raw_store.get_document(doc_id)
    if doc is None:
        return False

    if doc.status == "processed":
        return True

    result = extract([doc], embedder, chroma_store, source_hints=source_hints)
    if result.errors:
        return False

    raw_store.save_document(dataclasses.replace(doc, status="processed"))
    return True
