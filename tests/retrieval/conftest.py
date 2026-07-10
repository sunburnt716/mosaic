"""
Offline test fixtures for the retrieval-layer suite.

Provides a fake Groq-SDK-shaped chat client (no network, no `groq` import needed) and a
deterministic query embedder, so retrieval/router.py's contract can be exercised without
either external dependency installed.
"""

from __future__ import annotations

import json

import pytest

from tests.processing.conftest import fake_embedder  # noqa: F401 — re-exported as a fixture


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class FakeGroqClient:
    """Stand-in for a Groq SDK client: `.chat.completions.create(...)` returns a fixed reply.

    `reply` is whatever the model would have put in `message.content` — usually a JSON string,
    but tests can pass malformed text to exercise the parser's fallback path.
    """

    class _Completions:
        def __init__(self, reply: str):
            self._reply = reply
            self.last_kwargs: dict | None = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return _FakeCompletion(self._reply)

    class _Chat:
        def __init__(self, reply: str):
            self.completions = FakeGroqClient._Completions(reply)

    def __init__(self, reply: str):
        self.chat = FakeGroqClient._Chat(reply)


def make_groq_client(**payload) -> FakeGroqClient:
    """Build a FakeGroqClient whose reply is `payload` serialized as the model's JSON output."""
    return FakeGroqClient(json.dumps(payload))


@pytest.fixture
def fake_query_embedder():
    """Deterministic stand-in for the MiniLM query embedder — offline, no model download."""

    def _embed(text: str) -> list[float]:
        return [float(len(text)), float(sum(ord(c) for c in text) % 97)]

    return _embed
