"""
run.py — entrypoint, scheduler, and process lifecycle for the ingestion engine.

One-sentence contract: decide which sources are due right now, hand each due source
to the engine in isolation, sleep, and repeat — while owning the process's start/stop
lifecycle.

Explicitly NOT responsible for: fetching, parsing, normalising, dedup, embedding,
or storage. All per-source work lives behind engine.process_source(). run.py only
decides *when* and *who*, never *how*.

Hot-path wiring exception: main() (the composition root) is the one place in this
module that reaches past ingestion, building the closure ConcreteEngine calls after
storing a document for a `processing_mode: hot` source (see ingestion/engine.py's
ConcreteEngine docstring and CLAUDE.md's "Hot path wiring" for why this lives in main()
and not inside ConcreteEngine or elsewhere in ingestion/).

Scheduler design (tick-based):
  - The scheduler wakes every tick_interval seconds (e.g. 30 s).
  - On each wake it checks every enabled source against its poll_interval and
    last_polled_at to decide which are due.
  - Due sources are dispatched sequentially inside a try/except so one failure
    never aborts the rest (source isolation).
  - This fixed-tick design is intentionally simple. Computing the exact next-due
    timestamp to sleep precisely is a future optimisation; at the smallest expected
    poll_interval (~1 min), a 30 s tick wastes negligible CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ingestion.core.document import Document
from ingestion.core.source_config import SourceConfig, load_sources
from ingestion.engine import ConcreteEngine, Engine
from ingestion.storage.poll_state import PollStateStore
from ingestion.storage.raw_store import RawStore
from ingestion.storage.seen_store import SeenStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduling logic
# Pure functions: testable without a clock, network, or real engine.
# ---------------------------------------------------------------------------


def is_due(
    source: SourceConfig,
    last_polled_at: datetime | None,
    now: datetime,
) -> bool:
    """Return True if this source should be polled at `now`.

    A source is due when:
      - It has never been polled (last_polled_at is None), or
      - Enough time has passed since the last poll:
        now - last_polled_at >= source.poll_interval.

    Day/night interval logic (e.g. poll less frequently at night) lives here
    if it is ever added — nowhere else.

    Args:
        source:          the source to evaluate.
        last_polled_at:  UTC datetime of the most recent successful poll,
                         or None if the source has never been polled.
        now:             the scheduler's current UTC time (injected so callers
                         can test without touching the real clock).
    """
    if last_polled_at is None:
        return True
    elapsed = now - last_polled_at
    return elapsed >= source.poll_interval


def select_due_sources(
    sources: list[SourceConfig],
    poll_state: PollStateStore,
    now: datetime,
) -> list[SourceConfig]:
    """Return the subset of enabled sources that are due for polling at `now`.

    Disabled sources are excluded regardless of due-ness. They remain in the
    registry so re-enabling is a one-line YAML change, not a code change.

    Args:
        sources:     all sources from the registry (enabled and disabled).
        poll_state:  store to read each source's last_polled_at from.
        now:         the scheduler's current UTC time.
    """
    due: list[SourceConfig] = []
    for source in sources:
        if not source.enabled:
            log.debug("source_disabled_skipped", extra={"source": source.name})
            continue
        state = poll_state.get(source.name)
        last_polled_at = state.last_polled_at if state is not None else None
        if is_due(source, last_polled_at, now):
            due.append(source)
    return due


# ---------------------------------------------------------------------------
# Tick — one scheduler iteration
# ---------------------------------------------------------------------------


def tick(
    sources: list[SourceConfig],
    poll_state: PollStateStore,
    engine: Engine,
    now: datetime,
) -> None:
    """Run one scheduler iteration: find due sources and dispatch each in isolation.

    Source isolation guarantee: if engine.process_source() raises for one source,
    that failure is logged and the remaining due sources are still processed.
    A single bad source never aborts the tick or propagates upward.

    This function does not sleep. The caller owns timing so tick remains
    synchronous and fully unit-testable.

    Args:
        sources:    all sources from the registry.
        poll_state: store for reading last_polled_at (written by engine, not here).
        engine:     the pipeline orchestrator — called once per due source.
        now:        UTC time used for due-ness evaluation (injected for testability).
    """
    due = select_due_sources(sources, poll_state, now)
    log.info(
        "tick_start",
        extra={
            "due_count": len(due),
            "total_count": len(sources),
            "now": now.isoformat(),
        },
    )

    ok = 0
    failed = 0
    for source in due:
        try:
            engine.process_source(source)
            ok += 1
            log.info("source_processed", extra={"source": source.name})
        except Exception:
            failed += 1
            # Log at ERROR with the full traceback so operators can see what failed
            # without needing to reproduce the environment.
            log.exception("source_failed", extra={"source": source.name})

    log.info(
        "tick_done",
        extra={"due": len(due), "ok": ok, "failed": failed},
    )


# ---------------------------------------------------------------------------
# Long-lived loop
# ---------------------------------------------------------------------------


def run_forever(
    sources: list[SourceConfig],
    poll_state: PollStateStore,
    engine: Engine,
    tick_interval: float,
    stop_event: threading.Event,
) -> None:
    """Poll in a loop until stop_event is set.

    Wakes every tick_interval seconds, runs tick(), then sleeps again. The sleep
    is interruptible: stop_event.wait(timeout=...) returns immediately when
    stop_event fires, so shutdown does not wait out a full sleep cycle.

    Args:
        sources:       all sources from the registry.
        poll_state:    store for per-source last_polled_at.
        engine:        the pipeline orchestrator.
        tick_interval: seconds between scheduler wakes (e.g. 30.0). Distinct
                       from per-source poll_interval — the scheduler wakes
                       frequently and cheaply; most sources won't be due.
        stop_event:    set by the signal handler to trigger a clean shutdown.
    """
    log.info("scheduler_start", extra={"tick_interval_seconds": tick_interval})
    while not stop_event.is_set():
        now = datetime.now(tz=timezone.utc)
        tick(sources, poll_state, engine, now)
        # Interruptible sleep: returns early when stop_event is set so shutdown
        # doesn't block for up to tick_interval seconds.
        stop_event.wait(timeout=tick_interval)
    log.info("scheduler_stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.run",
        description=(
            "Mosaic ingestion scheduler. "
            "Polls sources on their configured intervals and feeds the pipeline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ingestion/config/sources.yaml"),
        metavar="PATH",
        help="Path to sources.yaml.",
    )
    parser.add_argument(
        "--tick-interval",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help=(
            "How often (in seconds) the scheduler wakes to check for due sources. "
            "Distinct from per-source poll_interval."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single tick then exit. Use this for cron / cloud-scheduler mode.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("data/chroma"),
        metavar="PATH",
        help=(
            "Directory for the persistent Chroma vector store, used for hot-path "
            "extraction. Only touched if at least one enabled source has "
            "processing_mode: hot."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser.parse_args(argv)


def _build_hot_path_callback(raw_store: RawStore, chroma_path: Path) -> Callable[[Document], None]:
    """Build the on_processed closure that extracts a document and marks it processed.

    Imports extraction.* lazily, here rather than at module top, so ingestion/run.py
    stays importable without chromadb/sentence-transformers installed when no source
    actually uses processing_mode: hot (matching this codebase's existing lazy-import
    convention for optional runtime deps).
    """
    import chromadb

    from extraction.chroma_store import ChromaVectorStore
    from extraction.embedder import MiniLMEmbedder
    from extraction.extraction_engine import extract

    embedder = MiniLMEmbedder()
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    chroma_store = ChromaVectorStore(chroma_client, embedder.model_name)

    def on_processed(doc: Document) -> None:
        result = extract(
            [doc], embedder, chroma_store, source_hints={doc.source_name: doc.doc_type}
        )
        if result.errors:
            raise RuntimeError(f"extraction failed for {doc.id}: {result.errors}")
        raw_store.save_document(dataclasses.replace(doc, status="processed"))

    return on_processed


def _configure_logging(level: str) -> None:
    """Configure structured JSON-line logging to stdout."""
    logging.basicConfig(
        level=getattr(logging, level),
        # One JSON object per line so log aggregators (Datadog, CloudWatch, etc.)
        # can parse fields without regex.
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
    """Parse CLI args, wire dependencies, and run the scheduler.

    Returns:
        0 on clean exit.
        1 on startup failure (bad config, missing file).
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    # --- Load and validate config BEFORE touching the network ---
    # Fail fast here; a misconfigured source should not be discoverable at poll time.
    try:
        sources = load_sources(args.config)
    except FileNotFoundError:
        log.error("config_not_found", extra={"path": str(args.config)})
        return 1
    except ValueError as exc:
        log.error("config_invalid", extra={"error": str(exc)})
        return 1

    enabled_count = sum(1 for s in sources if s.enabled)
    log.info(
        "sources_loaded",
        extra={"total": len(sources), "enabled": enabled_count},
    )

    # --- Wire dependencies ---
    # State/data files live alongside the config so they're easy to find and wipe for
    # a clean re-run.
    poll_state_path = args.config.parent / "poll_state.json"
    poll_state = PollStateStore(poll_state_path)
    raw_store = RawStore(str(args.config.parent / "raw.db"))
    seen_store = SeenStore(str(args.config.parent / "seen.db"))

    # Only wire the hot-path extraction callback (and its chromadb/sentence-transformers
    # deps) if at least one enabled source actually needs it.
    on_processed = None
    if any(s.enabled and s.processing_mode == "hot" for s in sources):
        on_processed = _build_hot_path_callback(raw_store, args.chroma_path)
        log.info("hot_path_enabled", extra={"chroma_path": str(args.chroma_path)})

    engine: Engine = ConcreteEngine(raw_store, seen_store, poll_state, on_processed)

    # --- Graceful shutdown ---
    # Signal sets stop_event. The current tick (or the current source within it)
    # finishes, the interruptible sleep returns early, and the loop exits cleanly.
    # --once and run_forever share this stop path so signal handling is consistent.
    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = str(signum)
        log.info("shutdown_signal_received", extra={"signal": sig_name})
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # --- Run ---
    if args.once:
        log.info("mode_once")
        tick(sources, poll_state, engine, datetime.now(tz=timezone.utc))
    else:
        run_forever(sources, poll_state, engine, args.tick_interval, stop_event)

    log.info("exit_clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
