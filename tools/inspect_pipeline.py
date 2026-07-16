"""
tools/inspect_pipeline.py — one-shot observability report for the whole pipeline.

Reads every moving part end-to-end and prints a stage-by-stage report so you can see,
in one place, what the pipeline actually did:

  1. SOURCES   — the registry (sources.yaml): tier, adapter, poll, doc_type,
                 processing_mode, enabled — i.e. what *should* flow, and how.
  2. RAW STORE — ingestion's output (ingestion/config/raw.db): normalized Documents
                 with their handoff status (unprocessed -> processed). This answers
                 "are the articles coming in what I want?"
  3. CHROMA    — extraction's output (data/chroma): chunks + the citation metadata
                 (source/tier/title/url/published_date) that must survive for synthesis.
  4. HANDOFF   — the seam between them: how many docs are still unprocessed (awaiting
                 extraction) vs embedded, so hot/cold coverage is visible at a glance.

This is a READ-ONLY diagnostic. It never fetches, normalizes, or writes anything — run
the pipeline first (see the two entry points below), then run this to inspect results:

    python -m ingestion.run  --once --chroma-path data/chroma   # ingest (+ hot extract)
    python -m extraction.run --once --chroma-path data/chroma   # cold-path backfill
    python -m tools.inspect_pipeline                            # <- this report

Nothing here is on the hot path; it's an operator tool, deliberately outside ingestion/
and extraction/ so it can read across both without coupling them.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Make `import ingestion` / `import extraction` resolve no matter how this file is
# launched. Run as a module (`python -m tools.inspect_pipeline`) the project root is
# already on sys.path; run as a script (`python tools/inspect_pipeline.py`, or the IDE
# "Run" button) Python puts tools/ on the path instead, so the project root — this
# file's parent's parent — must be added explicitly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# The extraction collection name is "<prefix><embedder model_name>"; keep in sync with
# extraction/chroma_store.py (_COLLECTION_PREFIX) and extraction/embedder.py (model_name).
_DEFAULT_COLLECTION = "mosaic_minilm-l6-v2"


# ---------------------------------------------------------------------------
# Small formatting helpers (ASCII-only so Windows consoles never choke)
# ---------------------------------------------------------------------------


def _rule(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def _sub(title: str) -> str:
    return f"\n-- {title} " + "-" * (74 - len(title))


def _trunc(text: str, width: int) -> str:
    """Truncate to width and collapse newlines so each record stays on one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def _counts(pairs: Counter) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(pairs.items())) or "(none)"


# ---------------------------------------------------------------------------
# Stage 1 — Sources registry
# ---------------------------------------------------------------------------


def _report_sources(config_path: Path) -> None:
    print(_rule("1. SOURCES  (registry: what should flow, and how)"))
    print(f"config: {config_path}")

    try:
        from ingestion.core.source_config import load_sources

        sources = load_sources(config_path)
    except FileNotFoundError:
        print("  !! config not found — nothing to report")
        return
    except ValueError as exc:
        print(f"  !! config invalid: {exc}")
        return

    enabled = [s for s in sources if s.enabled]
    hot = [s for s in enabled if s.processing_mode == "hot"]
    print(
        f"  {len(sources)} sources | {len(enabled)} enabled | "
        f"{len(hot)} enabled+hot (inline extraction on ingest)"
    )
    print(
        f"  {'name':<24} {'tier':>4} {'adapter':<10} {'poll':>7} "
        f"{'doc_type':<8} {'mode':<5} {'enabled':<7}"
    )
    for s in sources:
        poll = f"{int(s.poll_interval.total_seconds() // 60)}m"
        print(
            f"  {s.name:<24} {s.tier:>4} {s.adapter:<10} {poll:>7} "
            f"{s.doc_type:<8} {s.processing_mode:<5} {str(s.enabled):<7}"
        )


# ---------------------------------------------------------------------------
# Stage 2 — Raw store (ingestion output)
# ---------------------------------------------------------------------------


def _load_raw_docs(raw_db: Path) -> list[dict]:
    if not raw_db.exists():
        return []
    conn = sqlite3.connect(str(raw_db))
    try:
        rows = conn.execute("SELECT data FROM documents").fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) for r in rows]


def _report_raw_store(raw_db: Path, docs: list[dict], limit: int, source: str | None) -> None:
    print(_rule("2. RAW STORE  (ingestion output: normalized Documents)"))
    print(f"store: {raw_db}")
    if not raw_db.exists():
        print("  !! raw.db does not exist yet — run `python -m ingestion.run --once` first")
        return

    print(f"  {len(docs)} documents total")
    print(f"    by source: {_counts(Counter(d['source_name'] for d in docs))}")
    print(f"    by status: {_counts(Counter(d['status'] for d in docs))}")
    print(f"    by doc_type (advisory): {_counts(Counter(d['doc_type'] for d in docs))}")
    inferred = Counter(d.get('document_type') or '(not inferred)' for d in docs)
    print(f"    by document_type (inferred): {_counts(inferred)}")

    shown = [d for d in docs if source is None or d['source_name'] == source]
    print(_sub(f"sample documents (showing up to {limit})"))
    print(f"  {'status':<12} {'tier':>4} {'published':<11} {'source':<18} title")
    for d in shown[:limit]:
        print(
            f"  {d['status']:<12} {d['tier']:>4} {d['published_date'][:10]:<11} "
            f"{_trunc(d['source_name'], 18):<18} {_trunc(d['title'], 60)}"
        )


# ---------------------------------------------------------------------------
# Stage 3 — Chroma (extraction output)
# ---------------------------------------------------------------------------


def _report_chroma(
    chroma_path: Path, collection: str, limit: int, source: str | None
) -> list[dict]:
    print(_rule("3. CHROMA  (extraction output: chunks + citation metadata)"))
    print(f"store: {chroma_path}  collection: {collection}")
    if not chroma_path.exists():
        print("  !! chroma path does not exist yet — run extraction (hot or cold) first")
        return []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(collection)
    except Exception as exc:  # noqa: BLE001 — collection may not exist yet
        print(f"  !! could not open collection: {type(exc).__name__}: {exc}")
        return []

    got = col.get(include=["metadatas", "documents"])
    metas = got["metadatas"] or []
    text_by_id = dict(zip(got["ids"], got["documents"] or []))
    print(f"  {col.count()} chunks total")
    print(f"    by source: {_counts(Counter(m['source_name'] for m in metas))}")
    print(f"    by tier:   {_counts(Counter(str(m['tier']) for m in metas))}")

    shown_ids = [
        i for i, m in zip(got["ids"], metas) if source is None or m["source_name"] == source
    ]
    shown_meta = {i: m for i, m in zip(got["ids"], metas)}
    print(_sub(f"sample chunks with citation metadata (showing up to {limit})"))
    for cid in shown_ids[:limit]:
        m = shown_meta[cid]
        print(f"  [{_trunc(m['source_name'], 14)} t{m['tier']}] {_trunc(m['title'], 58)}")
        print(f"      url:  {_trunc(m['url'], 68)}")
        print(f"      when: {m['published_date']}   chunk_id: {cid}")
        print(f"      text: {_trunc(text_by_id.get(cid, ''), 66)}")
    return metas


# ---------------------------------------------------------------------------
# Stage 4 — Handoff seam
# ---------------------------------------------------------------------------


def _report_handoff(docs: list[dict], chroma_metas: list[dict]) -> None:
    print(_rule("4. HANDOFF  (ingestion -> extraction seam)"))
    if not docs:
        print("  (raw store empty — nothing to reconcile)")
        return

    unprocessed = [d for d in docs if d["status"] == "unprocessed"]
    processed = [d for d in docs if d["status"] == "processed"]
    docs_with_chunks = {m["document_id"] for m in chroma_metas}

    print(f"  raw store:  {len(processed)} processed, {len(unprocessed)} unprocessed")
    print(f"  chroma:     {len(docs_with_chunks)} distinct documents have chunks")
    # A processed doc with no chunks is normal (empty/too-short body -> 0 chunks);
    # an unprocessed doc that somehow has chunks would be a real inconsistency.
    processed_no_chunks = [d for d in processed if d["id"] not in docs_with_chunks]
    unprocessed_with_chunks = [d for d in unprocessed if d["id"] in docs_with_chunks]
    print(
        f"  processed-but-no-chunks: {len(processed_no_chunks)} "
        f"(expected for empty/short bodies)"
    )
    if unprocessed_with_chunks:
        print(
            f"  !! INCONSISTENT: {len(unprocessed_with_chunks)} unprocessed docs "
            f"already have chunks in chroma"
        )
    else:
        print("  consistency: OK (no unprocessed doc has stray chunks)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m tools.inspect_pipeline",
        description="Read-only, stage-by-stage report of the Mosaic pipeline's output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("ingestion/config/sources.yaml"))
    p.add_argument(
        "--raw-db",
        type=Path,
        default=None,
        help="Path to raw.db (defaults to <config dir>/raw.db).",
    )
    p.add_argument("--chroma-path", type=Path, default=Path("data/chroma"))
    p.add_argument("--collection", default=_DEFAULT_COLLECTION)
    p.add_argument("--source", default=None, help="Filter samples to one source name.")
    p.add_argument("--limit", type=int, default=8, help="Rows per sample section.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so non-ASCII titles/warnings print
    # cleanly instead of raising UnicodeEncodeError mid-report.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — older Pythons / redirected streams
        pass

    args = _parse_args(argv)
    raw_db = args.raw_db or (args.config.parent / "raw.db")

    print(_rule("MOSAIC PIPELINE INSPECTOR"))
    _report_sources(args.config)
    docs = _load_raw_docs(raw_db)
    _report_raw_store(raw_db, docs, args.limit, args.source)
    metas = _report_chroma(args.chroma_path, args.collection, args.limit, args.source)
    _report_handoff(docs, metas)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
