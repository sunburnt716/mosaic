"""
evals/run.py — run the answerability eval and print the buckets + headline rates.

Operator harness, counterpart to query/run.py and tools/inspect_pipeline.py. It loads the
labeled question set, runs each through the live read path against the existing Chroma
collection, prints a per-question table and a summary, and (optionally) writes a timestamped
JSON so before/after runs can be diffed as you add feeds.

    python -m evals.run                       # retrieval-only unless a Gemini key is set
    python -m evals.run --json evals/out/run-01.json

Router/synthesizer selection mirrors query/run.py: Groq routing when GROQ_API_KEY is set
(else offline embedding+profile, or --offline-router), Gemini synthesis when GEMINI_API_KEY
+ google-genai are present (else retrieval-only, and the full buckets need a re-run with a
key). Read-only: never fetches, extracts, or writes to the corpus. Log the summary to
Metrics.md afterward (see CLAUDE.md "Metrics").

GROQ_API_KEY / GEMINI_API_KEY are read from the environment; a `.env` file at the project
root is loaded automatically if present (see `.env.example`), same as query/run.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# override=False (default): a real environment variable always wins over the file.
load_dotenv(_PROJECT_ROOT / ".env")

_DEFAULT_COLLECTION = "mosaic_minilm-l6-v2"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run the answerability eval and report buckets + headline rates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--questions", type=Path, default=None, help="Path to questions.yaml.")
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument("--collection", default=_DEFAULT_COLLECTION)
    p.add_argument("--n-results", type=int, default=20)
    p.add_argument(
        "--offline-router",
        action="store_true",
        help="Skip Groq; route with query embedding + profile only.",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the full results + summary to this JSON file.",
    )
    return p.parse_args(argv)


def _gemini_available() -> bool:
    if not os.environ.get("GEMINI_API_KEY"):
        return False
    return importlib.util.find_spec("google.genai") is not None


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else ("-" if x is None else str(x))


def _pct(x) -> str:
    return f"{x * 100:.0f}%" if isinstance(x, float) else "n/a (needs synthesis)"


def _open_collection(chroma_path: Path, collection_name: str):
    if not chroma_path.exists():
        print(f"  !! chroma path does not exist: {chroma_path}")
        print("     Run ingestion + extraction first (see tools/inspect_pipeline.py).")
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        return client.get_collection(collection_name)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not open collection '{collection_name}': {type(exc).__name__}: {exc}")
        return None


def _print_table(results) -> None:
    print(
        f"\n{'id':<7} {'intent':<22} {'exp':<8} {'n':>3} {'top1':>6} {'top3':>6} "
        f"{'cite':>5} {'valid':>6}  bucket"
    )
    print("-" * 92)
    for r in results:
        cite = "-" if r.synthesis_citable is None else ("yes" if r.synthesis_citable else "no")
        valid = "-" if r.validator_passed is None else ("yes" if r.validator_passed else "no")
        print(
            f"{r.id:<7} {r.intent:<22} {r.expected:<8} {r.n_retrieved:>3} "
            f"{_fmt(r.top1_similarity):>6} {_fmt(r.top3_similarity):>6} "
            f"{cite:>5} {valid:>6}  {r.bucket}"
        )


def _print_summary(summary) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  total questions: {summary.total}   synthesis ran: {summary.synthesis_ran}")
    print(f"  buckets: {summary.bucket_counts}")
    print(
        f"  in-scope answered (working / in-scope): "
        f"{summary.in_scope_working}/{summary.in_scope_total}  "
        f"=> answerable-in-scope: {_pct(summary.answerable_in_scope_rate)}"
    )
    print(
        f"  out-of-scope declined (working / out-of-scope): "
        f"{summary.out_of_scope_declined}/{summary.out_of_scope_total}  "
        f"=> abstention: {_pct(summary.abstention_rate)}"
    )
    print(f"  avg top1 similarity (in-scope): {_fmt(summary.avg_top1_in_scope)}")
    if not summary.synthesis_ran:
        print("\n  NOTE: retrieval-only run — buckets are 'retrieval-only' and the headline")
        print("  rates need a re-run with GEMINI_API_KEY + google-genai to populate.")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    args = _parse_args(argv)

    from evals.harness import evaluate, load_questions, summarize
    from query.engine import OfflineRouter

    print("=" * 70)
    print("MOSAIC ANSWERABILITY EVAL")
    print("=" * 70)

    questions = load_questions(args.questions)
    print(f"  {len(questions)} labeled questions loaded")

    collection = _open_collection(args.chroma_path, args.collection)
    if collection is None:
        return 1

    use_groq = not args.offline_router and bool(os.environ.get("GROQ_API_KEY"))
    if use_groq:
        from retrieval.router import QueryRouter

        router = QueryRouter()
        print("  router: Groq (Llama 3.1 8B)")
    else:
        router = OfflineRouter()
        print(f"  router: offline ({'forced' if args.offline_router else 'no GROQ_API_KEY'})")

    synthesizer = None
    if _gemini_available():
        from generation.synthesizer import Synthesizer

        synthesizer = Synthesizer()
        print("  synthesis: Gemini Flash")
    else:
        print("  synthesis: skipped (no GEMINI_API_KEY / google-genai) — retrieval only")

    results = evaluate(
        questions,
        collection=collection,
        router=router,
        synthesizer=synthesizer,
        n_results=args.n_results,
    )
    summary = summarize(results)

    _print_table(results)
    _print_summary(summary)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_at": datetime.now(tz=timezone.utc).isoformat(),
            "collection": args.collection,
            "summary": dataclasses.asdict(summary),
            "results": [dataclasses.asdict(r) for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
