"""
query/run.py — CLI harness to ask the pipeline a question and see the whole read path run.

This is the operator counterpart to `ingestion/run.py` (write path) and
`tools/inspect_pipeline.py` (inspect): it reads the Chroma collection that ingestion +
extraction populated, runs `query.engine.answer()`, and prints routing, what was retrieved,
and the final cited answer.

    python -m query.run "What's the latest on inflation in Europe?"
    python -m query.run "Nvidia earnings" --tickers NVDA --n-results 8

Degrades on purpose so you can test with whatever you have configured:
  - Routing uses Groq (Llama 3.1 8B) when GROQ_API_KEY is set; otherwise it falls back to
    an offline router (query embedding + your --tickers/--sectors profile, no intent
    inference). Force the fallback with --offline-router.
  - Synthesis uses Gemini when GEMINI_API_KEY is set and google-genai is installed;
    otherwise it stops after retrieval and prints the retrieved context, telling you what
    to add for the full end-to-end path.

Read-only: never fetches, extracts, or writes. Run ingestion + extraction first (see the
two entry points in tools/inspect_pipeline.py), then ask a question here.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Resolve `import query` / `import retrieval` however this file is launched (module or
# script), mirroring tools/inspect_pipeline.py's path bootstrap.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Keep in sync with tools/inspect_pipeline.py and extraction/embedder.py's model_name.
_DEFAULT_COLLECTION = "mosaic_minilm-l6-v2"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m query.run",
        description="Ask the Mosaic pipeline a question and print the cited answer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("query", help="The question to ask, in quotes.")
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument("--collection", default=_DEFAULT_COLLECTION)
    p.add_argument(
        "--tickers",
        nargs="*",
        default=[],
        metavar="SYM",
        help="Profile tickers (uppercase, e.g. NVDA AMZN) — bias re-rank + filter search.",
    )
    p.add_argument(
        "--sectors",
        nargs="*",
        default=[],
        metavar="NAME",
        help="Profile sectors (lowercase) — backfills routing when the query names none.",
    )
    p.add_argument("--n-results", type=int, default=20, help="Max chunks to retrieve.")
    p.add_argument(
        "--offline-router",
        action="store_true",
        help="Skip Groq; route with query embedding + profile only (no intent inference).",
    )
    return p.parse_args(argv)


def _gemini_available() -> bool:
    """True only if both the key and the SDK are present — the two things a live call needs."""
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    return importlib.util.find_spec("google.genai") is not None


def _rule(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}"


def _open_collection(chroma_path: Path, collection_name: str):
    """Return the Chroma collection, or None with a printed reason if it can't be opened."""
    if not chroma_path.exists():
        print(f"  !! chroma path does not exist: {chroma_path}")
        print("     Run ingestion + extraction first:")
        print("       python -m ingestion.run  --once --chroma-path", chroma_path)
        print("       python -m extraction.run --once --chroma-path", chroma_path)
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        return client.get_collection(collection_name)
    except Exception as exc:  # noqa: BLE001 — collection may not exist yet
        print(f"  !! could not open collection '{collection_name}': {type(exc).__name__}: {exc}")
        print("     Has extraction written any chunks yet? Check: python -m tools.inspect_pipeline")
        return None


def _print_retrieval(result) -> None:
    ro = result.retrieval
    r = result.routing
    print(_rule("ROUTING"))
    print(
        f"  intent: {r.intent}   tickers: {r.tickers or '-'}   "
        f"sectors: {r.sectors or '-'}   window: {r.time_window_days}d"
    )

    print(_rule("RETRIEVAL"))
    print(
        f"  {ro.chunk_count} chunks in {len(ro.clusters)} clusters   "
        f"outlets: {', '.join(ro.outlets_represented) or '-'}"
    )
    print(
        f"  confidence (mean similarity): {ro.retrieval_confidence:.3f}   "
        f"citation_fields_present: {ro.citation_fields_present}"
    )
    for i, cluster in enumerate(ro.clusters[:8], 1):
        top = cluster.primary_chunk
        tick = f" [{top.ticker}]" if top.ticker else ""
        print(
            f"  {i}. ({cluster.corroboration}, {cluster.outlet_count} outlet(s)){tick} "
            f"{top.source_name}: {(top.text or '')[:70]}"
        )


def _print_answer(result) -> None:
    ans = result.answer
    print(_rule("ANSWER"))
    if ans is None:
        print("  (synthesis skipped — no Gemini configured)")
        print("  To generate a written answer, add the final stage:")
        print("    pip install google-genai   &&   set GEMINI_API_KEY=...")
        return

    print(f"  {ans.prose or '(no grounded claims — honest empty state)'}")
    if ans.confidence_warning:
        print(f"\n  ! {ans.confidence_warning}")
    if ans.citations:
        print("\n  Citations:")
        for c in ans.citations:
            print(f"    - {c.source}")
            print(f"      {c.url_with_fragment}")
    if ans.corroboration_summary:
        print(f"\n  Corroboration (outlets per story): {ans.corroboration_summary}")


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so non-ASCII source text prints cleanly.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — older Pythons / redirected streams
        pass

    args = _parse_args(argv)

    from query.engine import OfflineRouter, answer
    from retrieval.contracts import UserProfile

    print(_rule("MOSAIC QUERY"))
    print(f"  q: {args.query!r}")

    collection = _open_collection(args.chroma_path, args.collection)
    if collection is None:
        return 1

    profile = UserProfile(tickers=args.tickers, sectors=args.sectors)

    # --- Router: real Groq, or offline fallback ---
    use_groq = not args.offline_router and bool(os.environ.get("GROQ_API_KEY"))
    if use_groq:
        from retrieval.router import QueryRouter

        router = QueryRouter()
        print("  router: Groq (Llama 3.1 8B)")
    else:
        router = OfflineRouter()
        reason = "forced" if args.offline_router else "no GROQ_API_KEY"
        print(f"  router: offline embedding + profile ({reason})")

    # --- Synthesizer: real Gemini, or None (retrieval-only) ---
    synthesizer = None
    if _gemini_available():
        from generation.synthesizer import Synthesizer

        synthesizer = Synthesizer()
        print("  synthesis: Gemini Flash")
    else:
        print("  synthesis: skipped (no GEMINI_API_KEY / google-genai) — retrieval only")

    try:
        result = answer(
            args.query,
            profile,
            collection=collection,
            router=router,
            synthesizer=synthesizer,
            n_results=args.n_results,
        )
    except Exception as exc:  # noqa: BLE001 — surface any live-call failure legibly
        print(f"\n  !! query failed: {type(exc).__name__}: {exc}")
        return 1

    _print_retrieval(result)
    _print_answer(result)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
