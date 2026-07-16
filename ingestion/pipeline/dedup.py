"""Pure classification stage: which dedup level does an incoming Document trigger?

The three levels must NEVER be collapsed — each catches a different failure mode and
requires a different response from the engine. See the contract in test_dedup.py.

  classify(doc, seen_store, embedding=None) -> DedupResult

Priority is strict: L1 (content hash) > L2 (identity key) > L3 (embedding) > NEW.
  L1  exact bytes already seen            -> discard silently
  L2  same identity_key, different hash   -> article updated; ingest new version
  L3  no key match, embedding too similar -> same story, different outlet; ingest BOTH
  NEW no match at any level               -> ingest normally

This module is read-only w.r.t. the SeenStore and never calls an embedding model;
L3 compares pre-computed embeddings supplied by the caller.

`cosine_similarity` and `L3_SIMILARITY_THRESHOLD` are public (not `_`-prefixed) so the
Retrieval Pipeline's Phase 4 clustering can reuse them directly for cross-outlet corroboration
instead of duplicating the math or drifting onto a different threshold (see retrieval/cluster.py).
"""

import math
from enum import Enum

# Cosine-similarity threshold above which two documents are treated as the same story.
L3_SIMILARITY_THRESHOLD = 0.85


class DedupResult(Enum):
    NEW = "new"
    L1_DUPLICATE = "l1_duplicate"
    L2_UPDATE = "l2_update"
    L3_NEAR_DUPLICATE = "l3_near_duplicate"


def cosine_similarity(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def classify(doc, seen_store, embedding=None) -> DedupResult:
    # L1 — exact content match wins over everything (content unchanged, not an update).
    if seen_store.contains_hash(doc.content_hash):
        return DedupResult.L1_DUPLICATE

    # L2 — same logical article seen before, but the content differs: it was updated.
    stored_hash = seen_store.get_hash(doc.identity_key)
    if stored_hash is not None and stored_hash != doc.content_hash:
        return DedupResult.L2_UPDATE

    # L3 — no identity match, but semantically near an existing doc: cross-outlet story.
    if embedding is not None:
        for _doc_id, stored_embedding in seen_store.get_embeddings():
            if cosine_similarity(embedding, stored_embedding) >= L3_SIMILARITY_THRESHOLD:
                return DedupResult.L3_NEAR_DUPLICATE

    return DedupResult.NEW
