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
import re
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
    p.add_argument(
        "--trace",
        default=None,
        metavar="ID",
        help="Diagnose a single labeled question (e.g. pt-02): dump routing, retrieval, the "
        "raw Gemini output, parsed claims, and per-claim validation, then exit.",
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
        f"{'cite':>5} {'fb':>3}  {'bucket':<14} verdict"
    )
    print("-" * 100)
    for r in results:
        cite = "-" if r.synthesis_citable is None else ("yes" if r.synthesis_citable else "no")
        fb = "yes" if r.filter_fallback else "-"
        print(
            f"{r.id:<7} {r.intent:<22} {r.expected:<8} {r.n_retrieved:>3} "
            f"{_fmt(r.top1_similarity):>6} {_fmt(r.top3_similarity):>6} "
            f"{cite:>5} {fb:>3}  {r.bucket:<14} {r.verdict}"
        )


def _print_summary(summary) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if summary.citation_path_suspect:
        print("  " + "!" * 66)
        print(
            f"  !! CITATION PATH SUSPECT: {summary.strong_uncited_in_scope} in-scope "
            f"question(s) had strong retrieval (top1 >= floor)"
        )
        print("  !! but produced no citation. The headline rates below are NOT trustworthy")
        print(
            "  !! until the citation path is fixed. Diagnose with: python -m evals.run --trace <id>"
        )
        print("  " + "!" * 66)

    print(f"  total questions: {summary.total}   synthesis ran: {summary.synthesis_ran}")
    print(f"  buckets:  {summary.bucket_counts}")
    print(f"  verdicts: {summary.verdict_counts}")
    print(
        f"  answerable-in-scope (cited / in-scope): "
        f"{summary.in_scope_cited}/{summary.in_scope_total}  "
        f"=> {_pct(summary.answerable_in_scope_rate)}"
    )
    print(
        f"  meaningful abstention (declined / out-of-scope WITH strong retrieval): "
        f"{summary.out_of_scope_tempted_declined}/{summary.out_of_scope_tempted}  "
        f"=> {_pct(summary.meaningful_abstention_rate)}"
    )
    print(
        f"  filter-starvation: no-candidates={summary.no_candidates_count}  "
        f"filter-fallback-fired={summary.filter_fallback_count}"
    )
    print(f"  avg top1 similarity (in-scope): {_fmt(summary.avg_top1_in_scope)}")
    if not summary.synthesis_ran:
        print("\n  NOTE: retrieval-only run — citation-dependent buckets/rates are blank; only")
        print("  no-candidates + retrieval signal are observable. Re-run with GEMINI_API_KEY.")


_CHUNK_ID_RE = re.compile(r"^CHUNK_ID:\s*(\S+)", re.MULTILINE)


def _offered_chunk_ids(prompt: str | None) -> list[str]:
    """The CHUNK_IDs actually shown to Gemini — read from the assembled prompt verbatim."""
    return _CHUNK_ID_RE.findall(prompt or "")


def _validation_reason(vc) -> str:
    """Infer why a claim did/didn't ground from its validation_confidence (see validator.py)."""
    if vc.is_grounded and vc.validation_confidence >= 1.0:
        return "direct-hit (ID matched an offered chunk)"
    if vc.is_grounded:
        return f"semantic match ({vc.validation_confidence:.2f} >= 0.85)"
    if vc.validation_confidence > 0.0:
        return f"semantic below threshold ({vc.validation_confidence:.2f} < 0.85)"
    return "no match (ID absent AND no chunk >= 0.85 / no embeddings)"


def _diagnosis(result, offered: list[str], parsed_claims) -> str:
    """One-line pointer to the Phase B branch this question's failure implies."""
    from generation.synthesizer import INSUFFICIENT_DATA_MARKER

    raw = result.raw_synthesis or ""
    grounded = [c for c in (result.validated_claims or []) if c.is_grounded]
    if grounded:
        return "citation path WORKING for this question (>=1 grounded claim)."
    if raw.strip() == INSUFFICIENT_DATA_MARKER:
        return (
            "MARKER — the Gemini *call* failed (fail-closed). Check model id / SDK / key / quota."
        )
    if "CLAIM:" not in raw:
        return (
            "PROSE — Gemini didn't emit CLAIM/SOURCE_CHUNK_ID blocks. "
            "Check the prompt/parse contract."
        )
    cited_ids = [c.source_chunk_id for c in parsed_claims if c.source_chunk_id]
    if cited_ids and offered and not any(cid in offered for cid in cited_ids):
        return (
            "MANGLED IDs — cited IDs don't match any offered CHUNK_ID. "
            "Switch the prompt to short handles."
        )
    return (
        "IDs present but ungrounded — semantic fallback below 0.85; "
        "revisit the threshold / embeddings."
    )


def _trace_one(question, *, collection, router, synthesizer, n_results) -> int:
    from evals.harness import _similarity_at_ranks
    from generation.claim_parser import ClaimParser
    from query.engine import answer
    from retrieval.contracts import UserProfile

    result = answer(
        question.question,
        UserProfile(),
        collection=collection,
        router=router,
        synthesizer=synthesizer,
        n_results=n_results,
        trace=True,
    )
    n, top1, top3 = _similarity_at_ranks(result)
    offered = _offered_chunk_ids(result.prompt)

    print(f"\n{'=' * 70}\nTRACE — {question.id}\n{'=' * 70}")
    print(f"  question: {question.question!r}")
    print(f"  intent: {question.intent}   expected: {question.expected}")

    r = result.routing
    print(
        f"\n-- ROUTING --\n  intent: {r.intent}   tickers: {r.tickers or '-'}   "
        f"sectors: {r.sectors or '-'}   window: {r.time_window_days}d"
    )
    print(f"  filter_fallback: {result.filter_fallback}")

    print(f"\n-- RETRIEVAL --\n  n: {n}   top1: {_fmt(top1)}   top3: {_fmt(top3)}")
    print(f"  offered CHUNK_IDs ({len(offered)}):")
    for cid in offered:
        print(f"    {cid}")

    if result.raw_synthesis is None:
        print(
            "\n-- SYNTHESIS --\n  skipped (no Gemini). Re-run with GEMINI_API_KEY + "
            "google-genai to see the raw output, parsed claims, and validation."
        )
        return 0

    print(f"\n-- RAW GEMINI OUTPUT (verbatim) --\n{result.raw_synthesis}")

    parsed = ClaimParser().parse(result.raw_synthesis)
    print(f"\n-- PARSED CLAIMS ({len(parsed)}) --")
    for i, c in enumerate(parsed, 1):
        print(f"  [{i}] is_valid={c.is_valid}  source_chunk_id={c.source_chunk_id}")
        print(f"      claim: {(c.claim_text or '')[:90]}")

    validated = result.validated_claims or []
    print(f"\n-- VALIDATION ({len(validated)}) --")
    for i, vc in enumerate(validated, 1):
        print(
            f"  [{i}] grounded={vc.is_grounded}  support={vc.supporting_chunk_id}  "
            f"reason: {_validation_reason(vc)}"
        )

    print(f"\n-- DIAGNOSIS --\n  {_diagnosis(result, offered, parsed)}\n")
    return 0


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

    if args.trace:
        question = next((q for q in questions if q.id == args.trace), None)
        if question is None:
            print(f"  !! no question with id '{args.trace}' in the eval set")
            return 1
        return _trace_one(
            question,
            collection=collection,
            router=router,
            synthesizer=synthesizer,
            n_results=args.n_results,
        )

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
