<<<<<<< HEAD
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
=======
"""Orchestrator: wires every pipeline stage and drives one ingestion run.

The engine is the only place where side effects (fetch, store, seen-store update,
poll-state update) are coordinated. All pipeline stages (normalizer, dedup) are pure;
the engine decides when their results are committed.

  run(configs, raw_store, seen_store, poll_state_store) -> EngineResult

Source isolation: FetchError or NormalizationError from one source is caught and logged
without aborting remaining sources. Each source runs in its own error boundary.

Conditional-GET flow:
  1. Load poll state to build If-None-Match / If-Modified-Since headers.
  2. Merge those headers into the adapter call.
  3. If adapter raises NotModifiedSignal (304) → touch poll state, skip to next source.
  4. On 200 → run normalize→dedup→store, then update poll state with new validators.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ingestion.adapters.base import FetchError, NotModifiedSignal, TransportError
from ingestion.adapters.registry import get_adapter
from ingestion.pipeline.dedup import DedupResult, classify
from ingestion.pipeline.normalizer import NormalizationError, normalize
from ingestion.pipeline.quality import check as quality_check

log = logging.getLogger(__name__)


@dataclass
class SourceResult:
    source_name: str
    fetched: int = 0
    new: int = 0
    l1_duplicate: int = 0
    l2_update: int = 0
    l3_near_duplicate: int = 0
    errors: int = 0
    # per-record contract failures: bad record dropped, rest of batch kept (Phase 2)
    dropped_records: int = 0
    # whole batch refused fail-closed at the transport/parse layer (Phase 2)
    rejected_transport: bool = False
    skipped_304: bool = False
    quality_warnings: list = field(default_factory=list)
    # batch statistics computed by the quality gate (records, rates, unique counts)
    quality_stats: dict = field(default_factory=dict)


@dataclass
class EngineResult:
    sources: list[SourceResult] = field(default_factory=list)

    @property
    def total_new(self) -> int:
        return sum(s.new for s in self.sources)

    @property
    def total_errors(self) -> int:
        return sum(s.errors for s in self.sources)


def run(configs, raw_store, seen_store, poll_state_store) -> EngineResult:
    """Run one ingestion pass over all enabled configs."""
    result = EngineResult()

    for config in configs:
        if not config.enabled:
            continue

        src_result = SourceResult(source_name=config.name)
        result.sources.append(src_result)

        try:
            _process_source(config, raw_store, seen_store, poll_state_store, src_result)
        except Exception as exc:
            log.error("Unhandled error processing source %s: %s", config.name, exc)
            src_result.errors += 1

    return result


def _process_source(config, raw_store, seen_store, poll_state_store, src_result):
    poll_state = poll_state_store.get(config.name)
    cond_headers = poll_state.conditional_headers()
    merged_headers = {**config.headers, **cond_headers}
    patched_config = _with_headers(config, merged_headers)

    adapter_cls = get_adapter(config.adapter)
    adapter = adapter_cls()
    fetched_at = datetime.now(timezone.utc)

    try:
        items = list(adapter.fetch(patched_config))
    except NotModifiedSignal:
        log.debug("304 Not Modified for %s — skipping parse", config.name)
        poll_state_store.touch(config.name)
        src_result.skipped_304 = True
        return
    except TransportError as exc:
        # Fail-closed: a structurally-broken batch (HTML challenge page, malformed feed,
        # empty body) is refused whole — nothing reaches normalize/dedup/store.
        # Checked before FetchError because TransportError subclasses it.
        log.error("Transport validation rejected %s: %s", config.name, exc)
        src_result.errors += 1
        src_result.rejected_transport = True
        poll_state_store.touch(config.name)
        return
    except FetchError as exc:
        log.error("FetchError for %s: %s", config.name, exc)
        src_result.errors += 1
        poll_state_store.touch(config.name)
        return

    new_etag = None
    new_last_modified = None
    # --- Pass 1: normalize (per-record contract). Bad records drop-and-count; the rest
    # form the batch the quality gate and dedup operate on. ---
    batch = []
    for raw in items:
        src_result.fetched += 1
        # Adapters may embed validator headers in the raw dict for the engine to capture.
        new_etag = raw.pop("_etag", new_etag)
        new_last_modified = raw.pop("_last_modified", new_last_modified)

        try:
            doc = normalize(raw, config, fetched_at)
        except NormalizationError as exc:
            # Per-record contract failure: drop this record and keep the rest of the batch.
            log.warning("Dropping bad record from %s: %s", config.name, exc)
            src_result.dropped_records += 1
            continue

        log.debug(
            "normalized doc | source=%s id=%s url=%s published=%s title=%r body_len=%d tier=%d",
            doc.source_name,
            doc.id,
            doc.url,
            doc.published_date.isoformat(),
            doc.title,
            len(doc.body),
            doc.tier,
        )
        batch.append(doc)

    # --- Quality gate: runs on the full normalized batch BEFORE dedup, so collapse/
    # degeneracy signals aren't masked by deduplication. Advisory only — never drops. ---
    report = quality_check(batch, config)
    src_result.quality_warnings = report.warnings
    src_result.quality_stats = report.stats
    for w in report.warnings:
        log.warning("Quality gate [%s]: %s", config.name, w)

    # --- Pass 2: dedup + store. The gate's verdict does not change what is stored. ---
    for doc in batch:
        dedup_result = classify(doc, seen_store)

        if dedup_result == DedupResult.L1_DUPLICATE:
            src_result.l1_duplicate += 1

        elif dedup_result == DedupResult.L2_UPDATE:
            raw_store.save_raw(doc.id, doc.raw_payload)
            raw_store.save_document(doc)
            seen_store.set_hash(doc.identity_key, doc.content_hash)
            src_result.l2_update += 1
            log.debug("L2 update: %s", doc.identity_key)

        elif dedup_result == DedupResult.L3_NEAR_DUPLICATE:
            raw_store.save_raw(doc.id, doc.raw_payload)
            raw_store.save_document(doc)
            seen_store.set_hash(doc.identity_key, doc.content_hash)
            src_result.l3_near_duplicate += 1
            log.debug("L3 near-duplicate ingested: %s", doc.id)

        else:  # NEW
            raw_store.save_raw(doc.id, doc.raw_payload)
            raw_store.save_document(doc)
            seen_store.set_hash(doc.identity_key, doc.content_hash)
            src_result.new += 1

    poll_state_store.update(config.name, new_etag, new_last_modified)
    log.info(
        "Source %s: fetched=%d new=%d l1=%d l2=%d l3=%d dropped=%d errors=%d warnings=%d",
        config.name,
        src_result.fetched,
        src_result.new,
        src_result.l1_duplicate,
        src_result.l2_update,
        src_result.l3_near_duplicate,
        src_result.dropped_records,
        src_result.errors,
        len(report.warnings),
    )


def _with_headers(config, headers: dict):
    """Return a shallow copy of config with headers replaced."""
    from ingestion.core.source_config import SourceConfig

    return SourceConfig(
        name=config.name,
        adapter=config.adapter,
        tier=config.tier,
        url=config.url,
        enabled=config.enabled,
        params=config.params,
        headers=headers,
        poll_interval=config.poll_interval,
        transform=config.transform,
        expects=config.expects,
        max_fallback_title_rate=config.max_fallback_title_rate,
        max_empty_body_rate=config.max_empty_body_rate,
        min_records=config.min_records,
    )
>>>>>>> main
