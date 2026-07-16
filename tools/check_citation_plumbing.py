"""
tools/check_citation_plumbing.py — prove the citation ID chain is intact, without Gemini.

The citation path grounds a claim by looking up the `SOURCE_CHUNK_ID` the model emitted in a
dict keyed by `chunk_id` (query/engine.py's `chunks_by_id`). Post-fix, the model is shown a
short handle (S1, S2, …) that the engine translates back to the real `chunk_id` before that
lookup. This tool verifies, for every labelled eval question, that the whole chain lines up:

    prompt handle (S{n})  ->  real chunk_id  ->  IS a key in the validator's chunks_by_id
                                             ->  IS an exact id stored in Chroma

If every handle resolves into BOTH the validator dict AND Chroma with a byte-identical id,
the citation *plumbing* is provably intact — any zero-citation result is then a synthesis
issue (API failure, or a genuinely ungroundable answer), NOT an ID/format mismatch. This is
the disambiguation that separates "the citation code is broken" from "Gemini was down": it
needs no Gemini key and no network beyond the local MiniLM embedder, so it runs green even
when the API quota is exhausted.

This is a READ-ONLY diagnostic — it opens the existing Chroma collection and runs retrieval +
prompt assembly in memory; it never calls Gemini, never writes, never fetches. Run the
pipeline first (see tools/inspect_pipeline.py), then:

    python -m tools.check_citation_plumbing
    python -m tools.check_citation_plumbing --ids pt-02 tk-05   # just these questions
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Keep in sync with extraction/chroma_store.py (_COLLECTION_PREFIX) + extraction/embedder.py.
_DEFAULT_COLLECTION = "mosaic_minilm-l6-v2"
_DEFAULT_CHROMA_PATH = "data/chroma"


def _check_one(question, *, collection, router, now, n_results):
    """Return (n_handles, mismatches) for one question. A mismatch is a handle whose resolved
    chunk_id is missing from the validator dict or from Chroma (byte-identical)."""
    from generation.prompt_builder import PromptBuilder
    from retrieval.cluster import StoryClusterer
    from retrieval.contracts import UserProfile
    from retrieval.output import assemble_retrieval_output
    from retrieval.rerank import Ranker
    from retrieval.search import VectorSearch

    routing = router.route(question.question, UserProfile())
    retrieved = VectorSearch(collection).search(routing, n_results=n_results)
    ranked = Ranker().rank(retrieved, routing, now)
    clusters = StoryClusterer().cluster(ranked)
    assembled = PromptBuilder().build(
        assemble_retrieval_output(clusters), question.question, [], UserProfile()
    )
    # The exact dict the validator will key the model's SOURCE_CHUNK_ID against.
    chunks_by_id = {c.chunk_id: c for c in ranked}

    mismatches = []
    for handle, chunk_id in assembled.chunk_id_by_handle.items():
        in_validator = chunk_id in chunks_by_id
        got = collection.get(ids=[chunk_id])
        in_chroma = bool(got["ids"]) and got["ids"][0] == chunk_id
        if not (in_validator and in_chroma):
            mismatches.append((handle, chunk_id, in_validator, in_chroma))
    return len(assembled.chunk_id_by_handle), mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-path", default=_DEFAULT_CHROMA_PATH)
    parser.add_argument("--collection", default=_DEFAULT_COLLECTION)
    parser.add_argument("--ids", nargs="*", help="only these question ids (default: all)")
    parser.add_argument("--n-results", type=int, default=20)
    args = parser.parse_args(argv)

    import chromadb

    from evals.harness import load_questions
    from query.engine import OfflineRouter

    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(args.collection)
    # Offline router (local MiniLM) — no Groq key needed; routing only affects *which* chunks
    # are retrieved, and the ID chain holds for any retrieved set.
    router = OfflineRouter()
    now = datetime.now(tz=timezone.utc)

    questions = load_questions()
    if args.ids:
        wanted = set(args.ids)
        questions = [q for q in questions if q.id in wanted]

    print("\n" + "=" * 70)
    print(f"CITATION PLUMBING CHECK  (collection={args.collection}, no Gemini)")
    print("=" * 70)
    total_handles = 0
    total_bad = 0
    for q in questions:
        n_handles, mismatches = _check_one(
            q, collection=collection, router=router, now=now, n_results=args.n_results
        )
        total_handles += n_handles
        total_bad += len(mismatches)
        status = "OK" if not mismatches else f"MISMATCH x{len(mismatches)}"
        print(f"  {q.id:<7} handles={n_handles:>3}  -> {status}")
        for handle, chunk_id, in_validator, in_chroma in mismatches[:5]:
            print(
                f"      {handle}: chunk_id={chunk_id}  in_validator={in_validator}  "
                f"in_chroma={in_chroma}"
            )

    print(f"\n{'=' * 70}")
    if total_bad == 0:
        print(
            f"  PASS: all {total_handles} offered handles across {len(questions)} question(s) "
            f"resolve to a chunk_id present in BOTH the validator dict AND Chroma."
        )
        print("  => the citation PLUMBING is intact; zero-citation results are a synthesis")
        print("     issue (API health / ungroundable answer), not an ID/format mismatch.")
        return 0
    print(f"  FAIL: {total_bad}/{total_handles} handles did not resolve cleanly (see above).")
    print("  => a real ID/format mismatch exists between prompt, validator, and Chroma.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
