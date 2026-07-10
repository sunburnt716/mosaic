"""
Shared, fully-offline builders for retrieval-layer tests.

Mirrors tests/processing/fixtures.py's convention: sane defaults so individual tests only
override what they care about. Not a test module (no `test_` functions), so pytest does not
collect it; test files import these builders directly.
"""

from __future__ import annotations

from datetime import datetime, timezone

from retrieval.contracts import RetrievedChunk, RoutingResult, StoryCluster, UserProfile


def epoch(year: int, month: int, day: int, hour: int = 12) -> int:
    """UTC epoch seconds for a given date — convenience for recency-sensitive fixtures."""
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp())


def make_user_profile(**overrides) -> UserProfile:
    base = dict(tickers=[], sectors=[])
    base.update(overrides)
    return UserProfile(**base)


def make_routing_result(**overrides) -> RoutingResult:
    base = dict(
        intent="company_news",
        tickers=["NVDA"],
        sectors=["semiconductors"],
        time_window_days=30,
        query_embedding=[0.1, 0.2, 0.3],
    )
    base.update(overrides)
    return RoutingResult(**base)


def make_retrieved_chunk(**overrides) -> RetrievedChunk:
    base = dict(
        chunk_id="doc-0001#0",
        text="NVIDIA reported quarterly revenue ahead of analyst expectations.",
        source_name="Reuters",
        tier=1,
        published_epoch=epoch(2026, 7, 8),
        ticker="NVDA",
        similarity_score=0.85,
        url="https://example.com/article/0001",
        section_label=None,
        ordinal=0,
    )
    base.update(overrides)
    return RetrievedChunk(**base)


def make_story_cluster(**overrides) -> StoryCluster:
    primary = overrides.pop("primary_chunk", None) or make_retrieved_chunk()
    base = dict(
        cluster_id="doc-0001",
        chunks=[primary],
        outlet_count=1,
        corroboration="single",
        primary_chunk=primary,
    )
    base.update(overrides)
    return StoryCluster(**base)
