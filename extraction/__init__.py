"""Extraction stage — processing that sits after ingestion in the pipeline.

Reads `status: unprocessed` Documents from the ingestion raw store and produces processed
content. Phase 1 (chunking) is built: dispatch a Document to a type-specific strategy and
get back `Chunk`s carrying dual spans + citation provenance, ready for embedding.

  from extraction.engine import chunk_document
  chunks = chunk_document(document)

Embedding into Chroma and ticker/sector enrichment remain future phases. See README.md.
"""
