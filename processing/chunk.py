"""
The Chunk schema — Phase 1's output contract.

A Chunk is the unit the rest of the pipeline embeds (Phase 2), writes to Chroma (Phase 3),
retrieves, and cites. It carries everything two downstream roles need:

  - retrieval/embedding — the full chunk text and its `full_span` in the parent document
  - generation/citation — a surgical `highlight_span` (the excerpt shown back to the user
    with a clickable link) plus the provenance to cite it: source, url, tier, timestamp

Dual spans are intentional and load-bearing (see the Phase 1 spec): `full_span` gives the
synthesis model context, `highlight_span` gives it precision. Chunks are built for this dual
use from the start, never retrofitted.

Provenance fields are copied straight off the parent Document — they must survive into Chroma
so synthesis can always cite source + timestamp. Construction lives in `build_chunk` (and
`materialize_chunks`), keeping the copy in one place so no chunker repeats it — mirroring how
Document construction lives in the normalizer rather than on the dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.core.document import Document

# (start_char, end_char) into the parent document body.
Span = tuple[int, int]


@dataclass(frozen=True)
class Chunk:
    """Immutable chunk of a parent Document, ready for embedding and citation."""

    # --- identity ---
    chunk_id: str  # deterministic: parent document id + "#" + ordinal
    document_id: str  # parent Document.id, for traceability back to the raw store
    # --- content ---
    text: str
    full_span: Span  # whole chunk's bounds in the parent body (embedding/retrieval)
    highlight_span: Span  # surgical excerpt within the chunk (generation/citation)
    # --- provenance (copied from the parent Document; must reach Chroma for citation) ---
    title: str
    url: str
    source_name: str
    tier: int
    published_date: datetime
    # --- lifecycle ---
    chunked_at: str  # ISO 8601 timestamp of when this chunk was created


def _now_iso() -> str:
    """Current time as an ISO 8601 UTC string (chunk creation stamp)."""
    return datetime.now(timezone.utc).isoformat()


def build_chunk(
    document: "Document",
    ordinal: int,
    text: str,
    full_span: Span,
    highlight_span: Span,
    chunked_at: str | None = None,
) -> Chunk:
    """Assemble one Chunk, copying citation provenance off the parent Document.

    `ordinal` is the chunk's 0-based position within the document; combined with the parent
    id it yields the deterministic `chunk_id` (e.g. "8f3c2a…e1#0"). `chunked_at` is injectable
    so callers (and tests) can pin the stamp; it defaults to now (UTC).
    """
    return Chunk(
        chunk_id=f"{document.id}#{ordinal}",
        document_id=document.id,
        text=text,
        full_span=full_span,
        highlight_span=highlight_span,
        title=document.title,
        url=document.url,
        source_name=document.source_name,
        tier=document.tier,
        published_date=document.published_date,
        chunked_at=chunked_at or _now_iso(),
    )


def materialize_chunks(
    document: "Document",
    plans: list[tuple[Span, Span]],
    *,
    base: int = 0,
    start_ordinal: int = 0,
    chunked_at: str | None = None,
) -> list[Chunk]:
    """Turn strategy-produced span plans into Chunks with correct offsets and ordinals.

    Each plan is a `(full_span, highlight_span)` pair with spans *relative to the text the
    strategy planned over*. `base` shifts them into absolute parent-body coordinates (0 for a
    whole-body pass; the section start when a chunker splits a sub-section), and `start_ordinal`
    continues chunk numbering across sub-passes. The chunk text is sliced from the parent body
    by the absolute span, so it always matches `full_span` exactly.
    """
    chunks: list[Chunk] = []
    for offset, (full_span, highlight_span) in enumerate(plans):
        full = (base + full_span[0], base + full_span[1])
        highlight = (base + highlight_span[0], base + highlight_span[1])
        text = document.body[full[0] : full[1]]
        chunks.append(
            build_chunk(
                document,
                start_ordinal + offset,
                text,
                full,
                highlight,
                chunked_at=chunked_at,
            )
        )
    return chunks
