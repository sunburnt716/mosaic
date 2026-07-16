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

`ordinal` and `section_label` restore the metadata dependency flagged by the Retrieval Pipeline
spec: retrieval's citation phase needs both on every chunk's Chroma metadata. `ordinal` is the
same 0-based position already encoded in `chunk_id` ("{document_id}#{ordinal}"), now also stored
directly so callers don't have to parse it back out. `section_label` is the detected header text
for a section chunk (e.g. "RISK FACTORS"); it is `None` for chunkers with no header concept
(paragraph, fixed) or for a header-less preamble section.

`ticker` is copied from `document.tickers[0]` (or `None` if the enrichment step in
extraction_engine.py found none) — a deliberate simplification for a Document tagged with
multiple tickers, since `RetrievedChunk.ticker` (retrieval's own contract, locked) is a
single optional string, not a list. A multi-ticker article is filterable only by its
first-listed ticker today; splitting one Chunk across several tickers would need a real
schema change on the retrieval side, out of scope here.
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
    ordinal: int  # 0-based position within the parent document; also encoded in chunk_id
    # Copied from parent Document; used to evict stale Chroma chunks on L2 (updated article) writes.
    identity_key: str
    # --- content ---
    text: str
    full_span: Span  # whole chunk's bounds in the parent body (embedding/retrieval)
    highlight_span: Span  # surgical excerpt within the chunk (generation/citation)
    # --- provenance (copied from the parent Document; must reach Chroma for citation) ---
    title: str
    url: str
    source_name: str
    tier: int
    published_date: str
    # --- lifecycle ---
    chunked_at: str  # ISO 8601 timestamp of when this chunk was created
    # --- structure (section chunker only; None elsewhere) ---
    section_label: str | None = None  # detected header text, e.g. "RISK FACTORS"
    # --- enrichment (Phase 0 ticker extraction; None if no match) ---
    ticker: str | None = None  # document.tickers[0] — see module docstring for the caveat


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
    section_label: str | None = None,
) -> Chunk:
    """Assemble one Chunk, copying citation provenance off the parent Document.

    `ordinal` is the chunk's 0-based position within the document; combined with the parent
    id it yields the deterministic `chunk_id` (e.g. "8f3c2a…e1#0"). `chunked_at` is injectable
    so callers (and tests) can pin the stamp; it defaults to now (UTC). `section_label` is the
    detected header text for a section chunk; `None` for chunkers with no header concept.
    """
    return Chunk(
        chunk_id=f"{document.id}#{ordinal}",
        document_id=document.id,
        ordinal=ordinal,
        identity_key=document.identity_key,
        text=text,
        full_span=full_span,
        highlight_span=highlight_span,
        title=document.title,
        url=document.url,
        source_name=document.source_name,
        tier=document.tier,
        published_date=document.published_date.isoformat(),
        chunked_at=chunked_at or _now_iso(),
        section_label=section_label,
        ticker=document.tickers[0] if document.tickers else None,
    )


def materialize_chunks(
    document: "Document",
    plans: list[tuple[Span, Span]],
    *,
    base: int = 0,
    start_ordinal: int = 0,
    chunked_at: str | None = None,
    section_label: str | None = None,
) -> list[Chunk]:
    """Turn strategy-produced span plans into Chunks with correct offsets and ordinals.

    Each plan is a `(full_span, highlight_span)` pair with spans *relative to the text the
    strategy planned over*. `base` shifts them into absolute parent-body coordinates (0 for a
    whole-body pass; the section start when a chunker splits a sub-section), and `start_ordinal`
    continues chunk numbering across sub-passes. The chunk text is sliced from the parent body
    by the absolute span, so it always matches `full_span` exactly. `section_label` applies to
    every chunk produced by this call — correct because each call plans over a single section
    (or the whole body, where there is no section to label).
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
                section_label=section_label,
            )
        )
    return chunks
