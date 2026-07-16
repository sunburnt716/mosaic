"""
Contract tests for the retrieval dataclasses (retrieval/contracts.py).

Pins immutability (frozen, mirroring processing.chunk.Chunk) and that the citation
metadata-dependency fields (section_label/ordinal) default to None rather than being
required, so a RetrievedChunk can be built even when a source chunk predates the fix.
"""

from __future__ import annotations

import pytest

from tests.retrieval.fixtures import (
    make_retrieved_chunk,
    make_routing_result,
    make_story_cluster,
    make_user_profile,
)


class TestFrozenContracts:
    def test_user_profile_is_frozen(self):
        profile = make_user_profile()
        with pytest.raises(AttributeError):
            profile.tickers = ["NVDA"]

    def test_routing_result_is_frozen(self):
        routing = make_routing_result()
        with pytest.raises(AttributeError):
            routing.intent = "unknown"

    def test_retrieved_chunk_is_frozen(self):
        chunk = make_retrieved_chunk()
        with pytest.raises(AttributeError):
            chunk.similarity_score = 0.99

    def test_story_cluster_is_frozen(self):
        cluster = make_story_cluster()
        with pytest.raises(AttributeError):
            cluster.corroboration = "high"


class TestRetrievedChunkCitationFields:
    def test_citation_fields_default_absent(self):
        chunk = make_retrieved_chunk(section_label=None, ordinal=None)
        assert chunk.section_label is None
        assert chunk.ordinal is None

    def test_citation_fields_pass_through_when_present(self):
        chunk = make_retrieved_chunk(section_label="RISK FACTORS", ordinal=2)
        assert chunk.section_label == "RISK FACTORS"
        assert chunk.ordinal == 2

    def test_tier_present_but_not_a_score(self):
        # Tier is a visible label (spec's locked decision) — just a plain field, no scoring
        # logic lives on the contract itself.
        chunk = make_retrieved_chunk(tier=3)
        assert chunk.tier == 3
