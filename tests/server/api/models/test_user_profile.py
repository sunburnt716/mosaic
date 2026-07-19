"""Contract tests for server/api/models/userProfile.py's tickers field_validator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import shared.tickers as tickers_module
from server.api.models.userProfile import UserProfile


@pytest.fixture(autouse=True)
def seed_valid_tickers(monkeypatch):
    monkeypatch.setattr(tickers_module, "_valid_tickers", {"AAPL", "NVDA"})


def _make_profile(**overrides) -> UserProfile:
    now = datetime.now(tz=timezone.utc)
    defaults = dict(
        user_id="u1",
        username="alice",
        password_hash="hashed",
        created_at=now,
        updated_at=now,
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


class TestTickersValidation:
    def test_valid_tickers_pass(self):
        profile = _make_profile(tickers=["AAPL", "NVDA"])
        assert profile.tickers == ["AAPL", "NVDA"]

    def test_invalid_ticker_raises(self):
        with pytest.raises(ValidationError):
            _make_profile(tickers=["NOTREAL"])

    def test_lowercase_ticker_normalized_to_upper(self):
        profile = _make_profile(tickers=["aapl"])
        assert profile.tickers == ["AAPL"]

    def test_empty_tickers_default(self):
        profile = _make_profile()
        assert profile.tickers == []
