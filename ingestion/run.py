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
from ingestion.sources import _DEFAULT_INTERVAL, DEFAULT_REGISTRY_PATH, load_sources
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


def _sources_due(configs, poll_state_store, *, now=None) -> list[SourceConfig]:
    """Return configs whose next poll time has arrived."""
    if now is None:
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
