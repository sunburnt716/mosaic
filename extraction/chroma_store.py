"""
Phase 3 Chroma write — vectors + citation metadata, with stale-chunk management.

`ChromaVectorStore` is a thin wrapper around a `chromadb.ClientAPI` that enforces three
invariants the rest of the pipeline relies on:

  1. One model per collection — the collection name encodes the embedder's `model_name`
     slug (e.g. "mosaic_minilm-l6-v2").  Attempting to upsert into a collection that was
     created by a different model raises `ModelMismatchError` immediately so the bug is
     caught at write time, not silently at retrieval time.

  2. Cosine distance — the collection is always created with
     `configuration={"hnsw": {"space": "cosine"}}`.  MiniLM produces L2-normalised
     vectors; cosine is the correct similarity space.  This is immutable after the first
     write, so it must be set here.

  3. Stale chunk deletion on L2 update — when a document is updated (same `identity_key`,
     new `document_id`), the old Chroma chunks from the previous version are deleted
     before the new ones are upserted.  Without this, both versions stay in the index and
     retrieval may surface the stale copy.  The new chunks carry a fresh `chunked_at`
     timestamp, resetting the TTL window so they are not prematurely expired.

All metadata written to Chroma goes through `_to_metadata`, which converts every Chunk
field to a primitive type (str/int/float/bool) — ChromaDB rejects non-primitive values.
`Span` tuples are encoded as "start:end" strings and can be decoded by callers that need
the offsets (e.g. a highlight renderer).

Public surface:
  ModelMismatchError    — raised when embedder model does not match an existing collection
  ChromaVectorStore     — upsert/delete wrapper; collection creation happens lazily
  _to_metadata          — convert one Chunk to a Chroma-safe metadata dict (module-private;
                          exposed here for testing and for `extraction_engine.py`)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb

    from extraction.chunk import Chunk

_COLLECTION_PREFIX = "mosaic_"
# Metadata key stored on every Chroma collection to track which embedder created it.
# Used to detect model mismatches before the first write into a stale collection.
_MODEL_META_KEY = "embedder_model"


class ModelMismatchError(Exception):
    """Raised when an embedder tries to write into a collection built by a different model."""


def _to_metadata(chunk: "Chunk") -> dict[str, str | int | float | bool]:
    """Convert a Chunk to a flat dict of Chroma-safe primitive values.

    ChromaDB only accepts str, int, float, or bool as metadata values.  Span tuples are
    encoded as "start:end" strings so the offset information survives the round-trip.
    """
    metadata: dict[str, str | int | float | bool] = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "identity_key": chunk.identity_key,
        "ordinal": chunk.ordinal,
        "source_name": chunk.source_name,
        "url": chunk.url,
        "tier": chunk.tier,
        "published_date": chunk.published_date,
        # int seconds since epoch, derived from published_date — retrieval's
        # build_where_clause filters on this key (Chroma's $gte needs a sortable number,
        # not the ISO string) and RetrievedChunk.published_epoch reads it back.
        "published_epoch": int(datetime.fromisoformat(chunk.published_date).timestamp()),
        "title": chunk.title,
        "chunked_at": chunk.chunked_at,
        # Span tuples are not primitive — encode as "start:end" for round-trip fidelity.
        "full_span": f"{chunk.full_span[0]}:{chunk.full_span[1]}",
        "highlight_span": f"{chunk.highlight_span[0]}:{chunk.highlight_span[1]}",
    }
    # section_label is legitimately None for paragraph/fixed chunks (see CLAUDE.md,
    # Phase 1 decisions) — Chroma metadata values must be primitives, so omit the key
    # entirely rather than writing None; retrieval's metadata.get("section_label")
    # already treats a missing key the same as an explicit None.
    if chunk.section_label is not None:
        metadata["section_label"] = chunk.section_label
    # ticker is None whenever Phase 0's ticker enrichment found no match (most chunks,
    # today, since the registry is a starter set) — same omit-rather-than-write-None
    # rule as section_label. retrieval's build_where_clause filters on this key.
    if chunk.ticker is not None:
        metadata["ticker"] = chunk.ticker
    return metadata


class ChromaVectorStore:
    """Thin wrapper around a Chroma collection for Phase 3 vector writes.

    `client` is a `chromadb.ClientAPI` instance (persistent or ephemeral — the caller
    decides).  `embedder_model_name` is the slug from `Embedder.model_name`; it is
    encoded into the collection name and checked on every upsert.
    """

    def __init__(self, client: "chromadb.ClientAPI", embedder_model_name: str) -> None:
        self._client = client
        self._model_name = embedder_model_name
        self._collection = None

    @property
    def collection_name(self) -> str:
        return f"{_COLLECTION_PREFIX}{self._model_name}"

    def _get_or_create_collection(self):
        if self._collection is not None:
            return self._collection

        # Cosine distance is set here and is immutable after the first write.
        # MiniLM produces L2-normalised embeddings — cosine is the correct space.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={_MODEL_META_KEY: self._model_name},
            configuration={"hnsw": {"space": "cosine"}},
        )

        stored_model = self._collection.metadata.get(_MODEL_META_KEY)
        if stored_model != self._model_name:
            raise ModelMismatchError(
                f"Collection '{self.collection_name}' was created by model "
                f"'{stored_model}', not '{self._model_name}'. "
                "Use a separate collection or re-create the store with the correct model."
            )

        return self._collection

    def upsert(
        self,
        chunks: "list[Chunk]",
        embeddings: list[list[float]],
    ) -> None:
        """Write chunks and their embeddings to Chroma.

        Before upserting, any existing chunks for the same `identity_key` are deleted.
        This handles L2 document updates (same logical article, new version): the old
        chunks are evicted so retrieval never surfaces stale content.  Idempotent: if the
        same document version is re-processed, the delete is a no-op (the new chunk_ids
        match the old ones) and upsert overwrites in place.
        """
        if not chunks:
            return

        collection = self._get_or_create_collection()
        identity_key = chunks[0].identity_key

        # Delete any chunks belonging to a previous version of this logical document.
        # The where filter matches by identity_key regardless of document_id, so old
        # versions produced by L2 ingestion updates are evicted before the new ones land.
        existing = collection.get(where={"identity_key": {"$eq": identity_key}})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_to_metadata(c) for c in chunks],
        )
