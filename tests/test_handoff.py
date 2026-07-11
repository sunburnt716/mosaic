"""The ingestion -> extraction handoff boundary.

Ingestion's only output is normalized, deduped Documents written to the raw store stamped
`status: "unprocessed"`. The extraction stage (extraction/) reads them from there on its
own clock (cold path), or gets them handed inline via an injected callback right after
storage (hot path — see ingestion/engine.py's ConcreteEngine docstring). These tests pin
that contract:
  - the normalizer stamps freshly-ingested Documents `unprocessed`;
  - that status round-trips losslessly through the raw store;
  - a full engine run lands NEW Documents in the store as `unprocessed`;
  - ingestion's internal pipeline logic has NO import coupling to any downstream stage —
    except run.py, which lazily wires the hot-path callback as the composition root.
All offline.
"""

from datetime import datetime, timezone
from pathlib import Path

from ingestion.engine import ConcreteEngine
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

        status_field = next(f for f in dataclasses.fields(Document) if f.name == "status")
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
    def test_new_document_stored_as_unprocessed(self, monkeypatch, tmp_path):
        from ingestion.adapters.rss import RssAdapter

        seen, raw, poll = (
            SeenStore(":memory:"),
            RawStore(":memory:"),
            PollStateStore(tmp_path / "poll_state.json"),
        )
        try:
            item = load_fixture("rss_reuters_sample.json")
            monkeypatch.setattr(RssAdapter, "_fetch_feed", lambda self, url, headers: [item])
            config = make_source_config(name="reuters", adapter="rss", tier=1)

            ConcreteEngine(raw, seen, poll).process_source(config)

            # The stored Document is the engine's sole output and carries the handoff state.
            doc = normalize(item, config, datetime.now(timezone.utc))
            stored = raw.get_document(doc.id)
            assert stored is not None
            assert stored.status == _INGEST_STATUS
        finally:
            seen.close()
            raw.close()


# ---------------------------------------------------------------------------
# Boundary: ingestion does not couple to any downstream stage
# ---------------------------------------------------------------------------


class TestNoDownstreamCoupling:
    def test_ingestion_internals_import_no_downstream_stage(self):
        """Ingestion's internal pipeline logic must never import a downstream stage.

        The store is the only seam; the two halves run on different clocks. A stray
        import would couple them, so guard it by scanning ingestion package source —
        except run.py, which is the documented exception (see below).
        """
        ingestion_dir = Path(__file__).parent.parent / "ingestion"
        forbidden = ("processing", "extraction", "generation", "source_validation")
        offenders = []
        for py in ingestion_dir.rglob("*.py"):
            if py.name == "run.py":
                continue
            text = py.read_text(encoding="utf-8")
            for pkg in forbidden:
                if f"import {pkg}" in text or f"from {pkg}" in text:
                    offenders.append(f"{py.name} -> {pkg}")
        assert not offenders, f"ingestion must not import downstream stages: {offenders}"

    def test_run_py_extraction_import_is_lazy_not_module_level(self):
        """run.py (the composition root) is allowed to wire in extraction, but only

        lazily inside a function — never at module top-level. This is the one deliberate
        exception to the boundary above: ConcreteEngine takes an injected `on_processed`
        callback instead of importing extraction itself (see ConcreteEngine's docstring
        in ingestion/engine.py), and run.py's main() is what builds that callback. A
        top-level import here would make extraction.* a hard dependency of every
        ingestion run, even cold-path-only deployments with no chromadb/sentence-
        transformers installed.
        """
        run_py = Path(__file__).parent.parent / "ingestion" / "run.py"
        text = run_py.read_text(encoding="utf-8")
        module_level_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import extraction", "from extraction")):
                module_level_lines.append(line)
        # Only flag genuinely top-level (unindented) import statements; the lazy imports
        # inside _build_hot_path_callback are indented and must not trigger this.
        top_level_offenders = [line for line in module_level_lines if not line[0].isspace()]
        assert not top_level_offenders, (
            f"run.py must only import extraction lazily inside a function: {top_level_offenders}"
        )
