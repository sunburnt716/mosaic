"""Shared fixtures and builder functions for the ingestion test suite.

All tests import from here for consistent Document / SourceConfig construction.
Builders use sane defaults so individual tests only override what they care about.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture file from tests/fixtures/ by filename."""
    return json.loads((FIXTURES_DIR / name).read_text())


def load_fixture_bytes(name: str) -> bytes:
    """Load a fixture file's raw bytes (for XML/HTML payloads fed to adapters)."""
    return (FIXTURES_DIR / name).read_bytes()


class FakeResponse:
    """Minimal stand-in for requests.Response — enough for the adapters' fetch path.

    Lets adapter/gate tests drive `requests.get` offline (monkeypatch it to return one of
    these), exercising the real conditional-GET, transport, parse, and mapping code without
    touching the network.
    """

    def __init__(
        self,
        status_code=200,
        headers=None,
        content=b"",
        json_data=None,
        url="https://example.test/feed",
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self._json = json_data
        self.url = url

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def make_source_config(
    name: str = "test-reuters-rss",
    adapter: str = "rss",
    tier: int = 1,
    url: str = "https://feeds.reuters.com/reuters/topNews",
    enabled: bool = True,
    params: dict | None = None,
    headers: dict | None = None,
    doc_type: str = "article",
    transform: str | None = None,
    expects: dict | None = None,
    max_fallback_title_rate: float | None = None,
    max_empty_body_rate: float | None = None,
    min_records: int | None = None,
):
    """Build a SourceConfig for use in tests without touching config/sources.json."""
    from ingestion.core.source_config import SourceConfig

    return SourceConfig(
        name=name,
        adapter=adapter,
        tier=tier,
        url=url,
        enabled=enabled,
        params=params if params is not None else {"doc_type": doc_type},
        headers=headers or {},
        transform=transform,
        expects=expects or {},
        max_fallback_title_rate=max_fallback_title_rate,
        max_empty_body_rate=max_empty_body_rate,
        min_records=min_records,
    )


def make_document(
    id: str = "abc123def456",
    content_hash: str = "deadbeef" * 8,
    identity_key: str = "Reuters::newsml_L1N3D30GH",
    source_name: str = "Reuters",
    url: str = "https://reuters.com/markets/us/fed-cuts-rates-2024-01-15/",
    tier: int = 1,
    published_date: datetime | None = None,
    title: str = "Fed cuts rates by 25 basis points",
    body: str = "The Federal Reserve cut interest rates by 25 basis points.",
    doc_type: str = "article",
    raw_payload: dict | None = None,
    fetched_at: datetime | None = None,
    tickers: list | None = None,
    sectors: list | None = None,
    key_points: list | None = None,
    status: str = "ingested",
):
    """Build a fully-populated Document for use in tests."""
    from ingestion.core.document import Document

    return Document(
        id=id,
        content_hash=content_hash,
        identity_key=identity_key,
        source_name=source_name,
        url=url,
        tier=tier,
        published_date=published_date
        or datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        title=title,
        body=body,
        doc_type=doc_type,
        raw_payload=raw_payload or {"original": "payload"},
        fetched_at=fetched_at or datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc),
        tickers=tickers or [],
        sectors=sectors or [],
        key_points=key_points or [],
        status=status,
    )


# ---------------------------------------------------------------------------
# pytest fixtures (used via function argument injection)
# ---------------------------------------------------------------------------


@pytest.fixture
def reuters_rss_raw() -> dict:
    """Raw dict as an RSS adapter would yield for a Reuters article."""
    return load_fixture("rss_reuters_sample.json")


@pytest.fixture
def rest_json_raw() -> dict:
    """Raw dict as a REST JSON adapter would yield for a news API article."""
    return load_fixture("rest_json_sample.json")


@pytest.fixture
def reuters_source_config():
    """SourceConfig representing the Reuters RSS feed (tier 1, doc_type=article)."""
    return make_source_config()


@pytest.fixture
def edgar_source_config():
    """SourceConfig representing SEC EDGAR (tier 0, doc_type=filing, via getcurrent Atom)."""
    return make_source_config(
        name="sec-edgar",
        adapter="rss",
        tier=0,
        url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&output=atom",
        doc_type="filing",
        transform="edgar_filing_url",
        expects={"title": True, "url": True, "body": False},
    )


@pytest.fixture
def fetched_at() -> datetime:
    """Fixed UTC fetch timestamp for deterministic tests."""
    return datetime(2024, 1, 15, 15, 0, tzinfo=timezone.utc)
