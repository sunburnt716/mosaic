"""Tests for ingestion/storage/raw_store.py — RawStore.iter_unprocessed().

save_raw/save_document/get_document/get_raw are already exercised indirectly through
test_engine.py and test_handoff.py; this file covers iter_unprocessed(), the read side
of the ingestion -> extraction handoff that the extraction layer's hot/cold paths rely
on to find work.
"""

from __future__ import annotations

import dataclasses

import pytest

from ingestion.storage.raw_store import RawStore
from tests.conftest import make_document


@pytest.fixture
def raw():
    store = RawStore(":memory:")
    yield store
    store.close()


class TestIterUnprocessed:
    def test_empty_store_yields_nothing(self, raw):
        assert list(raw.iter_unprocessed()) == []

    def test_unprocessed_document_is_yielded(self, raw):
        doc = make_document(id="doc-1", status="unprocessed")
        raw.save_document(doc)

        results = list(raw.iter_unprocessed())

        assert len(results) == 1
        assert results[0].id == "doc-1"

    def test_processed_document_is_excluded(self, raw):
        doc = make_document(id="doc-1", status="processed")
        raw.save_document(doc)

        assert list(raw.iter_unprocessed()) == []

    def test_marking_processed_removes_it_from_next_call(self, raw):
        doc = make_document(id="doc-1", status="unprocessed")
        raw.save_document(doc)
        assert len(list(raw.iter_unprocessed())) == 1

        raw.save_document(dataclasses.replace(doc, status="processed"))

        assert list(raw.iter_unprocessed()) == []

    def test_mixed_batch_returns_only_unprocessed(self, raw):
        raw.save_document(make_document(id="doc-1", status="unprocessed"))
        raw.save_document(make_document(id="doc-2", status="processed"))
        raw.save_document(make_document(id="doc-3", status="unprocessed"))

        ids = {doc.id for doc in raw.iter_unprocessed()}

        assert ids == {"doc-1", "doc-3"}
