"""
Engine — the interface between the scheduler (run.py) and the ingestion pipeline.

The Engine Protocol is the seam that keeps scheduling logic decoupled from pipeline
implementation. run.py depends only on this Protocol; the concrete engine can be
developed and tested independently under its own spec.

Using typing.Protocol (structural subtyping) means the concrete engine does not need
to inherit from Engine — any object with a matching process_source method satisfies it.
This makes test stubs trivial to write without inheriting from or mocking this class.

Concrete engine responsibilities (documented here for context; implemented later):
  1. Resolve the adapter for the source via adapters/registry.py.
  2. Fetch raw items via adapter.fetch(source).
  3. For each raw item:
       a. normalize(raw, source, fetched_at) -> Document   (via pipeline/normalizer.py)
       b. classify(doc, seen_store)          -> DedupResult (via pipeline/dedup.py)
       c. Act on DedupResult:
            NEW          -> save_raw + save_document + set_hash
            L1_DUPLICATE -> discard silently
            L2_UPDATE    -> overwrite document + update hash
            L3_NEAR_DUP  -> save with cluster_id, preserving cross-outlet corroboration
  4. Write updated PollState to PollStateStore (last_polled_at, etag, last_modified).
  5. Log a per-source summary: fetched / new / l1 / l2 / l3 counts.

Source isolation at the per-article level is the concrete engine's responsibility;
source isolation at the per-source level is tick()'s responsibility in run.py.
"""
from __future__ import annotations

import logging
from typing import Protocol

from ingestion.core.source_config import SourceConfig

_log = logging.getLogger(__name__)


class Engine(Protocol):
    """Minimal interface that run.py depends on.

    Any object with a process_source method that matches this signature satisfies
    the Protocol — including test stubs and the eventual concrete implementation.
    No inheritance required.
    """

    def process_source(self, source: SourceConfig) -> None:
        """Run the full ingestion pipeline for one source.

        Must update PollStateStore with the new last_polled_at after a successful
        fetch so that run.py's is_due() check reflects the completed poll.

        Should raise only on unrecoverable source-level errors (e.g. adapter not
        registered, config structurally invalid for the adapter). Transient errors
        (network timeouts, individual malformed items) should be caught internally,
        logged, and counted — process_source should not raise for per-item failures.
        """
        ...


class PlaceholderEngine:
    """Satisfies the Engine Protocol with a logged no-op.

    Used until the concrete engine is implemented. Logs a warning per source so
    it is immediately obvious when this placeholder is active rather than silently
    doing nothing. Replace with ConcreteEngine once the engine spec is written.

    Note: because PlaceholderEngine does not update PollStateStore, sources will
    appear due on every tick while this placeholder is in use.
    """

    def process_source(self, source: SourceConfig) -> None:
        _log.warning(
            "engine_placeholder_noop",
            extra={
                "source": source.name,
                "hint": "ConcreteEngine is not yet implemented; no pipeline work was done.",
            },
        )
