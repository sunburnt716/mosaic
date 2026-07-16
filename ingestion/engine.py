"""
Engine — the interface between the scheduler (run.py) and the ingestion pipeline.

The Engine Protocol is the seam that keeps scheduling logic decoupled from pipeline
implementation. run.py depends only on this Protocol; the concrete engine can be
developed and tested independently under its own spec.

Using typing.Protocol (structural subtyping) means the concrete engine does not need
to inherit from Engine — any object with a matching process_source method satisfies it.
This makes test stubs trivial to write without inheriting from or mocking this class.

ConcreteEngine responsibilities:
  1. Resolve the adapter for the source via adapters/registry.py.
  2. Fetch raw items via adapter.fetch(source), with conditional-GET headers built from
     the previous PollState.
  3. For each raw item:
       a. normalize(raw, source, fetched_at) -> Document   (via pipeline/normalizer.py)
       b. classify(doc, seen_store)          -> DedupResult (via pipeline/dedup.py)
       c. Act on DedupResult:
            NEW          -> save_raw + save_document + set_hash
            L1_DUPLICATE -> discard silently
            L2_UPDATE    -> overwrite document + update hash
            L3_NEAR_DUP  -> save, preserving cross-outlet corroboration
  4. Write updated PollState to PollStateStore (last_polled_at, etag, last_modified).
  5. Log a per-source summary: fetched / new / l1 / l2 / l3 counts.

Source isolation at the per-article level is the concrete engine's responsibility (a bad
record is dropped-and-counted, not fatal); source isolation at the per-source level is
tick()'s responsibility in run.py — process_source deliberately does not catch unexpected
exceptions itself, only the fetch-boundary signals it knows how to handle.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol

from ingestion.adapters.base import FetchError, NotModifiedSignal, TransportError
from ingestion.adapters.registry import get_adapter
from ingestion.core.document import Document
from ingestion.core.source_config import SourceConfig
from ingestion.pipeline.body_enrichment import FetchUrl, default_fetch_url, enrich_body
from ingestion.pipeline.dedup import DedupResult, classify
from ingestion.pipeline.normalizer import NormalizationError, normalize
from ingestion.pipeline.quality import check as quality_check
from ingestion.storage.poll_state import PollState, PollStateStore
from ingestion.storage.raw_store import RawStore
from ingestion.storage.seen_store import SeenStore

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

    Kept as a test/bootstrap stand-in for ConcreteEngine below (e.g. when no store
    wiring is wanted). Logs a warning per source so it is immediately obvious when
    this placeholder is active rather than silently doing nothing.

    Note: because PlaceholderEngine does not update PollStateStore, sources will
    appear due on every tick while this placeholder is in use.
    """

    def process_source(self, source: SourceConfig) -> None:
        _log.warning(
            "engine_placeholder_noop",
            extra={
                "source": source.name,
                "hint": "PlaceholderEngine is in use; no pipeline work was done.",
            },
        )


@dataclass
class SourceResult:
    """Per-source outcome of one process_source call, for structured logging."""

    source_name: str
    fetched: int = 0
    new: int = 0
    l1_duplicate: int = 0
    l2_update: int = 0
    l3_near_duplicate: int = 0
    errors: int = 0
    # per-record contract failures: bad record dropped, rest of batch kept
    dropped_records: int = 0
    # hot-path extraction failures (per-document; extraction itself is isolated so
    # these never abort the source, mirroring dropped_records)
    extraction_errors: int = 0
    # whole batch refused fail-closed at the transport/parse layer
    rejected_transport: bool = False
    skipped_304: bool = False
    quality_warnings: list = field(default_factory=list)
    # batch statistics computed by the quality gate (records, rates, unique counts)
    quality_stats: dict = field(default_factory=dict)


def _conditional_headers(state: PollState | None) -> dict[str, str]:
    """Build If-None-Match / If-Modified-Since headers from a stored PollState."""
    if state is None:
        return {}
    headers: dict[str, str] = {}
    if state.etag:
        headers["If-None-Match"] = state.etag
    if state.last_modified:
        headers["If-Modified-Since"] = state.last_modified
    return headers


class ConcreteEngine:
    """The real Engine: wires adapter -> normalizer -> dedup -> storage for one source.

    All side effects (fetch, store, seen-store update, poll-state update) are
    coordinated here; the pipeline stages themselves (normalizer, dedup) are pure.

    `on_processed` is the hot-path extraction seam. This module must never import
    `extraction.*` directly (see tests/test_handoff.py's TestNoDownstreamCoupling — the
    raw store is meant to be the only seam between ingestion and extraction, since the
    two run on different clocks). So instead of calling extraction code here, a stored
    document for a `processing_mode: hot` source is handed to this injected callback.
    `ingestion/run.py`'s `main()` (the composition root) is the one place that actually
    imports `extraction.extraction_engine.extract` and builds the closure passed in
    here; ConcreteEngine itself stays agnostic to what "processed" means downstream.
    """

    def __init__(
        self,
        raw_store: RawStore,
        seen_store: SeenStore,
        poll_state_store: PollStateStore,
        on_processed: Callable[[Document], None] | None = None,
        body_fetcher: FetchUrl = default_fetch_url,
    ) -> None:
        self._raw_store = raw_store
        self._seen_store = seen_store
        self._poll_state_store = poll_state_store
        self._on_processed = on_processed
        # Injected so enrichment fetches (e.g. EDGAR filing bodies) stay offline-testable;
        # only sources with SourceConfig.body_fetch set ever trigger it.
        self._body_fetcher = body_fetcher

    def process_source(self, source: SourceConfig) -> None:
        result = SourceResult(source_name=source.name)
        self._process(source, result)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process(self, config: SourceConfig, result: SourceResult) -> None:
        poll_state = self._poll_state_store.get(config.name)
        cond_headers = _conditional_headers(poll_state)
        merged_headers = {**config.headers, **cond_headers}
        patched_config = dataclasses.replace(config, headers=merged_headers)

        adapter_cls = get_adapter(config.adapter)  # ConfigError propagates: fatal, unrecoverable
        adapter = adapter_cls()
        fetched_at = datetime.now(timezone.utc)

        try:
            items = list(adapter.fetch(patched_config))
        except NotModifiedSignal:
            _log.debug("304 Not Modified for %s — skipping parse", config.name)
            self._update_poll_state(config.name, fetched_at)
            result.skipped_304 = True
            return
        except TransportError as exc:
            # Fail-closed: a structurally-broken batch (HTML challenge page, malformed
            # feed, empty body) is refused whole — nothing reaches normalize/dedup/store.
            # Checked before FetchError because TransportError subclasses it.
            _log.error("Transport validation rejected %s: %s", config.name, exc)
            result.errors += 1
            result.rejected_transport = True
            self._update_poll_state(config.name, fetched_at)
            return
        except FetchError as exc:
            _log.error("FetchError for %s: %s", config.name, exc)
            result.errors += 1
            self._update_poll_state(config.name, fetched_at)
            return

        new_etag = None
        new_last_modified = None
        # --- Pass 1: normalize (per-record contract). Bad records drop-and-count; the
        # rest form the batch the quality gate and dedup operate on. ---
        batch = []
        for raw in items:
            result.fetched += 1
            # Adapters may embed validator headers in the raw dict for the engine to capture.
            new_etag = raw.pop("_etag", new_etag)
            new_last_modified = raw.pop("_last_modified", new_last_modified)

            # Body enrichment (opt-in per source): fetch the real page text and replace the
            # feed snippet BEFORE normalize, so content_hash/document_id reflect it. Best-
            # effort — a failed fetch returns the record unchanged (keeps the summary).
            if config.body_fetch:
                raw = enrich_body(raw, config, fetch_url=self._body_fetcher)

            try:
                doc = normalize(raw, config, fetched_at)
            except NormalizationError as exc:
                # Per-record contract failure: drop this record and keep the rest of the batch.
                _log.warning("Dropping bad record from %s: %s", config.name, exc)
                result.dropped_records += 1
                continue

            _log.debug(
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

        # --- Quality gate: runs on the full normalized batch BEFORE dedup, so
        # collapse/degeneracy signals aren't masked by deduplication. Advisory only. ---
        report = quality_check(batch, config)
        result.quality_warnings = report.warnings
        result.quality_stats = report.stats
        for w in report.warnings:
            _log.warning("Quality gate [%s]: %s", config.name, w)

        # --- Pass 2: dedup + store. The gate's verdict does not change what is stored. ---
        for doc in batch:
            dedup_result = classify(doc, self._seen_store)

            if dedup_result == DedupResult.L1_DUPLICATE:
                result.l1_duplicate += 1
                continue

            self._raw_store.save_raw(doc.id, doc.raw_payload)
            self._raw_store.save_document(doc)
            self._seen_store.set_hash(doc.identity_key, doc.content_hash)

            if dedup_result == DedupResult.L2_UPDATE:
                result.l2_update += 1
                _log.debug("L2 update: %s", doc.identity_key)
            elif dedup_result == DedupResult.L3_NEAR_DUPLICATE:
                result.l3_near_duplicate += 1
                _log.debug("L3 near-duplicate ingested: %s", doc.id)
            else:  # NEW
                result.new += 1

            # --- Hot path: extract inline for processing_mode="hot" sources. Isolated
            # per document so an extraction failure never aborts the rest of the batch,
            # matching the per-record isolation used for NormalizationError above. ---
            if config.processing_mode == "hot" and self._on_processed is not None:
                try:
                    self._on_processed(doc)
                except Exception:
                    result.extraction_errors += 1
                    _log.exception("Hot-path extraction failed for %s/%s", config.name, doc.id)

        self._update_poll_state(config.name, fetched_at, new_etag, new_last_modified)
        _log.info(
            "Source %s: fetched=%d new=%d l1=%d l2=%d l3=%d dropped=%d errors=%d "
            "extraction_errors=%d warnings=%d",
            config.name,
            result.fetched,
            result.new,
            result.l1_duplicate,
            result.l2_update,
            result.l3_near_duplicate,
            result.dropped_records,
            result.errors,
            result.extraction_errors,
            len(report.warnings),
        )

    def _update_poll_state(
        self,
        source_name: str,
        now: datetime,
        new_etag: str | None = None,
        new_last_modified: str | None = None,
    ) -> None:
        """Bump last_polled_at to `now`, preserving prior validators when no new one arrived.

        Used both for a full processed batch (new_etag/new_last_modified from the response)
        and for a 304/error short-circuit (called with no new validators, which preserves
        whatever was stored — a "touch").
        """
        previous = self._poll_state_store.get(source_name)
        etag = new_etag if new_etag is not None else (previous.etag if previous else None)
        last_modified = (
            new_last_modified
            if new_last_modified is not None
            else (previous.last_modified if previous else None)
        )
        self._poll_state_store.set(
            source_name,
            PollState(last_polled_at=now, etag=etag, last_modified=last_modified),
        )
