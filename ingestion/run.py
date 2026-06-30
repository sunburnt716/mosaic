<<<<<<< HEAD
"""
run.py — entrypoint, scheduler, and process lifecycle for the ingestion engine.

One-sentence contract: decide which sources are due right now, hand each due source
to the engine in isolation, sleep, and repeat — while owning the process's start/stop
lifecycle.

Explicitly NOT responsible for: fetching, parsing, normalising, dedup, embedding,
or storage. All per-source work lives behind engine.process_source(). run.py only
decides *when* and *who*, never *how*.

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
import logging
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from ingestion.core.source_config import SourceConfig, load_sources
from ingestion.engine import Engine, PlaceholderEngine
from ingestion.storage.poll_state import PollStateStore

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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser.parse_args(argv)


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
    # Poll state lives alongside the config so it's easy to find and wipe for a
    # clean re-run. The concrete engine will be wired here once implemented.
    poll_state_path = args.config.parent / "poll_state.json"
    poll_state = PollStateStore(poll_state_path)
    engine: Engine = PlaceholderEngine()

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
=======
"""Entrypoint and scheduler for the ingestion engine.

Usage:
  python -m ingestion.run [--config PATH] [--once] [--source NAME] [--log-level LEVEL]

  --config      Path to the source registry JSON (default: config/sources.json)
  --once        Run a single pass then exit (no scheduling loop)
  --source      Run only the named source (useful for debugging a single feed)
  --log-level   Python logging level: DEBUG | INFO | WARNING | ERROR (default: INFO)

In scheduled mode the loop wakes every 60 s, checks which sources are due
(last_polled_at + poll_interval <= now), and runs the engine for those sources only.
Each source has its own cadence via poll_interval in config/sources.json so sources
don't march in lockstep. SIGINT/SIGTERM are caught and trigger a clean shutdown after
the current source finishes.
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ingestion.core.source_config import SourceConfig
from ingestion.engine import run as engine_run
from ingestion.sources import DEFAULT_REGISTRY_PATH, _DEFAULT_INTERVAL, load_sources
from ingestion.storage.poll_state import PollStateStore
from ingestion.storage.raw_store import RawStore
from ingestion.storage.seen_store import SeenStore

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"

_shutdown = False


def _parse_interval(s: str) -> timedelta:
    """Parse a simple interval string like '5m', '1h', '30s' into a timedelta."""
    units = {"s": 1, "m": 60, "h": 3600}
    s = s.strip().lower()
    unit = s[-1]
    if unit not in units:
        raise ValueError(f"Unknown interval unit {s!r}. Use s/m/h (e.g. '10m').")
    return timedelta(seconds=int(s[:-1]) * units[unit])


def _sources_due(configs, poll_state_store) -> list[SourceConfig]:
    """Return configs whose next poll time has arrived."""
    now = datetime.now(timezone.utc)
    due = []
    for config in configs:
        if not config.enabled:
            continue
        state = poll_state_store.get(config.name)
        if state.last_polled_at is None:
            due.append(config)
            continue
        interval = _parse_interval(config.poll_interval or _DEFAULT_INTERVAL)
        if now >= state.last_polled_at + interval:
            due.append(config)
    return due


def _print_result(result) -> None:
    for src in result.sources:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": src.source_name,
            "fetched": src.fetched,
            "new": src.new,
            "l1_duplicate": src.l1_duplicate,
            "l2_update": src.l2_update,
            "l3_near_duplicate": src.l3_near_duplicate,
            "dropped_records": src.dropped_records,
            "rejected_transport": src.rejected_transport,
            "errors": src.errors,
            "skipped_304": src.skipped_304,
            "quality_warnings": src.quality_warnings,
            "quality_stats": src.quality_stats,
        }
        print(json.dumps(record), flush=True)


def main(argv=None) -> int:
    global _shutdown

    parser = argparse.ArgumentParser(description="Mosaic ingestion engine")
    parser.add_argument("--config", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--source", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    configs = load_sources(args.config)
    if args.source:
        configs = [c for c in configs if c.name == args.source]
        if not configs:
            log.error("No source named %r found in %s", args.source, args.config)
            return 1

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen_store = SeenStore(str(_DATA_DIR / "seen.db"))
    raw_store = RawStore(str(_DATA_DIR / "raw.db"))
    poll_state_store = PollStateStore(str(_DATA_DIR / "poll_state.db"))

    def _handle_signal(sig, frame):
        global _shutdown
        log.info("Shutdown signal received — finishing current source then exiting.")
        _shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        if args.once:
            result = engine_run(configs, raw_store, seen_store, poll_state_store)
            _print_result(result)
            return 0 if result.total_errors == 0 else 1

        log.info("Starting scheduled ingestion loop. Ctrl-C to stop.")
        while not _shutdown:
            due = _sources_due(configs, poll_state_store)
            if due:
                result = engine_run(due, raw_store, seen_store, poll_state_store)
                _print_result(result)
            time.sleep(60)

    finally:
        seen_store.close()
        raw_store.close()
        poll_state_store.close()

>>>>>>> main
    return 0


if __name__ == "__main__":
    sys.exit(main())
