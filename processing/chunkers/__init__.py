"""
Chunking strategies, dispatched by document type.

Each strategy is a pure function `chunk_*(document, **config) -> list[Chunk]` (no I/O).
`registry.get_chunker(document_type)` picks the right one; `fixed` is the fallback for tweets,
unknown, and unstructured text. Strategies share span-planning internals so the section
chunker can reuse paragraph/fixed logic when a section overflows.
"""
