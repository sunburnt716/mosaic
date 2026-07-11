"""
extraction/run.py — standalone CLI entrypoint for cold-path extraction backfill.

Loads the source registry, wires MiniLM + Chroma, then drives the Phase 0→1→2→3
chain (type inference → chunking → embedding → Chroma write) over every `unprocessed`
Document sitting in the raw store.

This is the batch-mode counterpart to the two trigger paths described in CLAUDE.md
"Extraction layer":

  Hot path  — wired into ingestion/run.py's scheduler: a `processing_mode: hot` source
              gets extracted inline, right after ingestion/engine.py's ConcreteEngine
              stores each Document (see ConcreteEngine's `on_processed` callback).

  Cold path — this CLI is one caller of the cold path: it sweeps
              RawStore.iter_unprocessed() and calls extraction.cold_path.ensure_processed()
              for each one. It's useful both for `processing_mode: cold` sources that
              were never hot-processed, and as a catch-up run for any hot-path
              extraction failures (ConcreteEngine counts but doesn't retry those). The
              other cold-path caller — a query-time cache-miss in a future `retrieval/`
              layer — doesn't exist yet; `ensure_processed()` is ready for it either way.

Both paths share the same `extract()` function via `ensure_processed()`. The split is a
calling convention, not a code fork.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m extraction.run",
        description=(
            "Mosaic extraction pipeline. Embeds normalised Documents and writes vectors to Chroma."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ingestion/config/sources.yaml"),
        metavar="PATH",
        help="Path to sources.yaml (same registry used by ingestion/run.py).",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("data/chroma"),
        metavar="PATH",
        help="Directory for the persistent Chroma vector store.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one batch then exit. (Future: omit for a continuous loop.)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format=(
            '{"time":"%(asctime)s",'
            '"level":"%(levelname)s",'
            '"logger":"%(name)s",'
            '"message":"%(message)s"}'
        ),
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )


def main(argv: list[str] | None = None) -> int:
    """Wire dependencies and run one cold-path backfill sweep.

    Returns:
        0 on clean exit.
        1 on startup failure (bad config, missing file).
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    # --- Load and validate source registry ---
    from ingestion.core.source_config import load_sources

    try:
        sources = load_sources(args.config)
    except FileNotFoundError:
        log.error("config_not_found", extra={"path": str(args.config)})
        return 1
    except ValueError as exc:
        log.error("config_invalid", extra={"error": str(exc)})
        return 1

    log.info("sources_loaded", extra={"total": len(sources)})

    # --- Wire extraction dependencies ---
    import chromadb

    from extraction.chroma_store import ChromaVectorStore
    from extraction.cold_path import ensure_processed
    from extraction.embedder import MiniLMEmbedder
    from ingestion.storage.raw_store import RawStore

    embedder = MiniLMEmbedder()
    chroma_client = chromadb.PersistentClient(path=str(args.chroma_path))
    chroma_store = ChromaVectorStore(chroma_client, embedder.model_name)
    raw_store = RawStore(str(args.config.parent / "raw.db"))

    log.info(
        "extraction_wired",
        extra={
            "embedder": embedder.model_name,
            "chroma_collection": chroma_store.collection_name,
            "chroma_path": str(args.chroma_path),
        },
    )

    # --- Sweep every unprocessed Document in the raw store ---
    # source_hints maps source_name → doc_type for Phase 0 type-inference advisory.
    source_hints = {s.name: s.doc_type for s in sources if s.doc_type}

    processed = 0
    failed = 0
    for doc in raw_store.iter_unprocessed():
        if ensure_processed(doc.id, raw_store, embedder, chroma_store, source_hints=source_hints):
            processed += 1
        else:
            failed += 1
            log.warning("cold_path_extraction_failed", extra={"doc_id": doc.id})

    summary = {"documents_processed": processed, "documents_failed": failed}
    print(json.dumps(summary, indent=2))

    log.info(
        "extraction_done",
        extra={"documents_processed": processed, "documents_failed": failed},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
