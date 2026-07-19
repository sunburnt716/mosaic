"""Local cache of valid SEC-registered ticker symbols, used to validate ticker input
during user-profile creation (server/api/models/userProfile.py) and LLM-extracted ticker
signal during query routing (retrieval/router.py).

Refreshed on a schedule via refresh_tickers_cli.py — never called from a request path.
Sourced from SEC's free company_tickers.json (no API key, no quota), republished nightly
at ~3am ET, so a daily refresh cadence matches the source's own update schedule.
"""

from __future__ import annotations

from typing import Callable

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_DEFAULT_TIMEOUT_SECONDS = 10

_valid_tickers: set[str] = set()

FetchJson = Callable[[str, dict], dict]


def _default_fetch_json(url: str, headers: dict) -> dict:
    """Fetch `url` and return its parsed JSON body. Lazy-imports requests (only live
    refreshes need it), mirroring ingestion/pipeline/body_enrichment.py's fetch_url."""
    import requests

    response = requests.get(url, headers=headers, timeout=_DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def refresh_tickers(user_agent_email: str, *, fetch_json: FetchJson = _default_fetch_json) -> None:
    """Fetch the current SEC ticker list and replace the in-memory set.

    `user_agent_email` is sent as the User-Agent header — SEC silently rejects requests
    without one. `fetch_json` is injectable so tests never hit the network.
    """
    data = fetch_json(SEC_TICKERS_URL, {"User-Agent": user_agent_email})
    global _valid_tickers
    _valid_tickers = {entry["ticker"].upper() for entry in data.values()}


def is_valid_ticker(ticker: str) -> bool:
    return ticker.upper() in _valid_tickers
