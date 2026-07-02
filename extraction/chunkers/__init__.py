"""Chunking strategies, dispatched by document type.

Each strategy is a pure function `chunk_*(document, **config) -> list[Chunk]` (no I/O).
`registry.get_chunker(doc_type)` picks the right one; `fixed` is the fallback for
unstructured text. Strategies share span-planning internals so the section chunker can
reuse paragraph/fixed logic when a section overflows.
"""
