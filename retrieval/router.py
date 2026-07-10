"""
Phase 1 — Query Router: freeform query text -> structured signal + query embedding.

The routing output becomes the *constraints* every downstream retrieval phase filters and
scores against (metadata filter in Phase 2, profile bias in Phase 3). Two calls happen here,
kept deliberately separate:

  - classification + extraction — Llama 3.1 8B on Groq, prompted for JSON-only output
    (no prose, no fences), reusing the structured-output discipline generation will need.
  - query embedding — the shared MiniLM embedder (processing.utils.embedding), the *same*
    model the corpus is embedded with. Mixing embedding models between query and corpus is
    forbidden (CLAUDE.md's collection invariant); this module never picks its own model.

Both clients are lazily constructed and injectable so the offline unit suite never imports
`groq` or downloads MiniLM (mirrors the ingestion adapters' lazy `requests`/`feedparser`
imports and processing.utils.tokenization's lazy tokenizer).

Non-goals (per spec): no multi-step query planning or agentic decomposition, no query
rewriting/expansion.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from processing.utils.embedding import embed_text
from retrieval.contracts import RoutingResult, UserProfile

VALID_INTENTS = {"earnings_deep_dive", "sector_trend", "company_news", "unknown"}
DEFAULT_TIME_WINDOW_DAYS = 30

_GROQ_MODEL = "llama-3.1-8b-instant"

_SYSTEM_PROMPT = """You classify investing-news queries. Given a user query, respond with ONLY \
a JSON object (no prose, no code fences) with exactly these keys:
  "intent": one of "earnings_deep_dive", "sector_trend", "company_news", "unknown"
  "tickers": list of uppercase ticker symbols mentioned or clearly implied, e.g. ["NVDA"]
  "sectors": list of lowercase sector names mentioned or clearly implied, e.g. ["semiconductors"]
  "time_window_days": integer days of relevant history implied by the query (default 30)
If nothing is confidently extractable, use empty lists and intent "unknown"."""


def _parse_classification(raw_content: str) -> dict[str, Any]:
    """Parse the model's JSON response, falling back to 'unknown' on malformed output.

    The model is instructed to return JSON only, but nothing upstream validates that — a
    malformed or refused response degrades to the same safe default as a genuinely
    unclassifiable query, rather than raising and failing the whole request.
    """
    try:
        parsed = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


class QueryRouter:
    """Phase 1: turn a freeform query into a RoutingResult."""

    def __init__(
        self,
        client: Any = None,
        embedder: Callable[[str], list[float]] = embed_text,
    ):
        """`client` is a Groq-SDK-shaped client (`.chat.completions.create(...)`); injectable
        for tests. `embedder` defaults to the shared MiniLM query embedder."""
        self._client = client
        self._embedder = embedder

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        from groq import Groq

        return Groq(api_key=os.environ["GROQ_API_KEY"])

    def _classify(self, query: str) -> dict[str, Any]:
        client = self._resolve_client()
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return _parse_classification(response.choices[0].message.content)

    def route(self, query: str, profile: UserProfile) -> RoutingResult:
        """Classify + extract `query`, backfilling tickers/sectors from `profile` when empty."""
        raw = self._classify(query)

        intent = raw.get("intent")
        if intent not in VALID_INTENTS:
            intent = "unknown"

        tickers = [t for t in (raw.get("tickers") or []) if isinstance(t, str)]
        sectors = [s for s in (raw.get("sectors") or []) if isinstance(s, str)]
        if not tickers:
            tickers = list(profile.tickers)
        if not sectors:
            sectors = list(profile.sectors)

        time_window_days = raw.get("time_window_days")
        if not isinstance(time_window_days, int) or time_window_days <= 0:
            time_window_days = DEFAULT_TIME_WINDOW_DAYS

        return RoutingResult(
            intent=intent,
            tickers=tickers,
            sectors=sectors,
            time_window_days=time_window_days,
            query_embedding=self._embedder(query),
        )
