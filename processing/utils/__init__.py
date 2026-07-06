"""
Shared, strategy-agnostic utilities for Phase 1 chunking.

One home for the machinery every chunker needs — the MiniLM tokenizer, section-header
detection, and highlight selection — so no chunker reimplements them and the embedding
model is never loaded in more than one place.
"""
