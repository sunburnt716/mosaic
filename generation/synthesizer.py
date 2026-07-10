"""
Phase 2 — Gemini Synthesis: call Gemini Flash with the assembled prompt, produce structured
CLAIM/SOURCE_CHUNK_ID/CONFIDENCE text.

The client is lazily constructed and injectable (mirrors retrieval.router.QueryRouter's Groq
pattern), so the offline unit suite never imports `google.genai` or makes a network call.

**Fail-closed retry policy**: every exception from a *call* to the client — transient or not,
since this module has no reliable way to distinguish the two without depending on a specific
SDK's exception hierarchy — is retried with exponential backoff up to `max_attempts` times. If
every attempt fails, `synthesize()` returns `INSUFFICIENT_DATA_MARKER` rather than raising or
returning a partial/garbled response. That marker is deliberately *not* valid
CLAIM/SOURCE_CHUNK_ID/CONFIDENCE text — fed into Phase 3's ClaimParser it parses to zero
*valid* claims (it may still surface as a single `is_valid=False` block, since ClaimParser
never silently drops non-empty text — see that module), which Phase 4's existing "zero claims
survive" rejection policy already turns into the honest empty-state answer either way. No
separate failure-signaling path between phases is needed.

This fail-closed guarantee covers *call* failures only. Resolving the client in the first place
(missing `GEMINI_API_KEY`, the `google-genai` package not installed) is a configuration error,
not API flakiness — retrying an identically-misconfigured client would never succeed, so that
failure propagates immediately rather than being swallowed into a silent "insufficient data".
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

GEMINI_MODEL = "gemini-2.0-flash"
TEMPERATURE = 0.3
MAX_OUTPUT_TOKENS = 800

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_BACKOFF_SECONDS = 1.0

INSUFFICIENT_DATA_MARKER = "INSUFFICIENT_DATA: synthesis failed after retries"


class Synthesizer:
    """Phase 2: prompt string -> Gemini's raw structured-text response (or a fail-closed marker)."""

    def __init__(
        self,
        client: Any = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_backoff_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        """`client` is a google-genai-SDK-shaped client
        (`.models.generate_content(model=, contents=, config=) -> response` with `.text`);
        injectable for tests. `sleep_fn` is injectable so tests never actually sleep."""
        self._client = client
        self._max_attempts = max_attempts
        self._base_backoff_seconds = base_backoff_seconds
        self._sleep_fn = sleep_fn

    def _resolve_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google import genai

        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def synthesize(self, prompt: str) -> str:
        """Call Gemini Flash with `prompt`, retrying on failure.

        Never raises for a failed *call* (falls back to `INSUFFICIENT_DATA_MARKER`); does
        raise if the client itself can't be resolved (missing API key, SDK not installed) —
        see module docstring.
        """
        client = self._resolve_client()

        for attempt in range(self._max_attempts):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config={
                        "temperature": TEMPERATURE,
                        "max_output_tokens": MAX_OUTPUT_TOKENS,
                    },
                )
                return response.text
            except Exception:
                is_last_attempt = attempt == self._max_attempts - 1
                if is_last_attempt:
                    break
                self._sleep_fn(self._base_backoff_seconds * (2**attempt))

        return INSUFFICIENT_DATA_MARKER
