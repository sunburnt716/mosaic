"""Phase 5 — the ingestion -> processing handoff boundary.

Ingestion's only output is normalized, deduped Documents written to the raw store stamped
`status: "unprocessed"`. The (future) processing stage reads them from there on its own clock.
These tests pin that contract:
  - the normalizer stamps freshly-ingested Documents `unprocessed`;
  - that status round-trips losslessly through the raw store;
  - a full engine run lands NEW Documents in the store as `unprocessed`;
  - ingestion has NO import coupling to any downstream stage.
All offline.
"""

from datetime import datetime, timezone
from pathlib import Path

from ingestion.engine import run
from ingestion.pipeline.normalizer import normalize
from ingestion.storage.poll_state import PollStateStore
from ingestion.storage.raw_store import RawStore
from ingestion.storage.seen_store import SeenStore
from tests.conftest import load_fixture, make_document, make_source_config

_INGEST_STATUS = "unprocessed"


# ---------------------------------------------------------------------------
# The normalizer stamps the handoff status
# ---------------------------------------------------------------------------


class TestNormalizerStampsUnprocessed:
    def test_normalized_document_is_unprocessed(
        self, reuters_rss_raw, reuters_source_config, fetched_at
    ):
        doc = normalize(reuters_rss_raw, reuters_source_config, fetched_at)
        assert doc.status == _INGEST_STATUS

    def test_document_default_status_is_unprocessed(self):
        # The contract's canonical initial state (the normalizer relies on this default).
        import dataclasses

        from ingestion.core.document import Document

        status_field = next(
            f for f in dataclasses.fields(Document) if f.name == "status"
        )
        assert status_field.default == _INGEST_STATUS


# ---------------------------------------------------------------------------
# The raw store round-trips status losslessly
# ---------------------------------------------------------------------------


class TestRawStoreRoundTripsStatus:
    def test_status_survives_save_and_get(self):
        raw = RawStore(":memory:")
        try:
            doc = make_document(status=_INGEST_STATUS)
            raw.save_document(doc)
            restored = raw.get_document(doc.id)
            assert restored.status == _INGEST_STATUS
        finally:
            raw.close()


# ---------------------------------------------------------------------------
# A full engine run hands off unprocessed Documents
# ---------------------------------------------------------------------------


class TestEngineHandoff:
    def test_new_document_stored_as_unprocessed(self, monkeypatch):
        from ingestion.adapters.rss import RssAdapter

        seen, raw, poll = (
            SeenStore(":memory:"),
            RawStore(":memory:"),
            PollStateStore(":memory:"),
        )
        try:
            item = load_fixture("rss_reuters_sample.json")
            monkeypatch.setattr(
                RssAdapter, "_fetch_feed", lambda self, url, headers: [item]
            )
            config = make_source_config(name="reuters", adapter="rss", tier=1)

            result = run([config], raw, seen, poll)
            assert result.sources[0].new == 1

            # The stored Document is the engine's sole output and carries the handoff state.
            doc = normalize(item, config, datetime.now(timezone.utc))
            stored = raw.get_document(doc.id)
            assert stored is not None
            assert stored.status == _INGEST_STATUS
        finally:
            seen.close()
            raw.close()
            poll.close()


# ---------------------------------------------------------------------------
# Boundary: ingestion does not couple to any downstream stage
# ---------------------------------------------------------------------------


class TestNoDownstreamCoupling:
    def test_ingestion_imports_no_downstream_stage(self):
        """Ingestion must never import extraction/generation/source_validation.

        The store is the only seam; the two halves run on different clocks. A stray import
        would couple them, so guard it by scanning the ingestion package source.
        """
        ingestion_dir = Path(__file__).parent.parent / "ingestion"
        forbidden = ("extraction", "generation", "source_validation")
        offenders = []
        for py in ingestion_dir.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            for pkg in forbidden:
                if f"import {pkg}" in text or f"from {pkg}" in text:
                    offenders.append(f"{py.name} -> {pkg}")
        assert not offenders, (
            f"ingestion must not import downstream stages: {offenders}"
        )
