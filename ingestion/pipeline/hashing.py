"""Pure functions for the three identifiers every Document must carry.

All functions here are deterministic and side-effect-free: same input always yields
the same output, no I/O, no state. See the contract in test_hashing.py.

  content_hash(raw_body)        -> SHA-256 hex of whitespace-normalized content (L1 dedup)
  identity_key(source, art_id)  -> "{source}::{art_id}" stable across updates (L2 dedup)
  document_id(identity, hash)   -> globally unique id; changes when content changes
"""

import hashlib


def content_hash(raw_body: str | bytes) -> str:
    """SHA-256 hex of the whitespace-normalized content.

    Leading/trailing whitespace is stripped and internal runs (including newlines)
    are collapsed to single spaces before hashing, so trivial formatting differences
    don't produce spurious distinct hashes. Two documents with identical content_hash
    are byte-level duplicates (L1).
    """
    text = raw_body.decode("utf-8") if isinstance(raw_body, bytes) else raw_body
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def identity_key(source_name: str, source_article_id: str) -> str:
    """Stable, human-readable key identifying the same logical article across updates.

    Composed as "{source_name}::{source_article_id}" where source_article_id is the
    source's own primary key (RSS guid, EDGAR accession number, etc.). Drives L2:
    a seen key with a *different* content_hash means the article was updated.
    """
    return f"{source_name}::{source_article_id}"


def document_id(identity_key: str, content_hash: str) -> str:
    """Globally unique Document.id derived from (identity_key, content_hash).

    Stable for a specific version of an article; changes if either input changes.
    The two inputs are joined with a NUL separator so that no pair of distinct
    (identity_key, content_hash) inputs can collide by concatenation ambiguity.
    Returns a 64-char lowercase hex string — safe as a Chroma document ID.
    """
    return hashlib.sha256(f"{identity_key}\x00{content_hash}".encode()).hexdigest()
