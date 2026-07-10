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


class FakeChromaCollection:
    """Stand-in for a chromadb.Collection: `.query(...)` returns a fixed canned response.

    `response` is the raw dict shape a real Collection.query() returns (ids/distances/
    metadatas/documents, one inner list per query in the batch). Records the last call's
    kwargs so tests can assert the where-clause/n_results actually sent.
    """

    def __init__(self, response: dict):
        self._response = response
        self.last_kwargs: dict | None = None

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


def make_query_response(
    ids: list[str],
    distances: list[float],
    metadatas: list[dict],
    documents: list[str],
    embeddings: list[list[float]] | None = None,
) -> dict:
    """Build a single-query Chroma query() response from parallel per-chunk lists.

    `embeddings` omitted (None) simulates a caller that didn't request them in `include=`,
    matching how VectorSearch must degrade when a collection/response lacks vectors.
    """
    response = {
        "ids": [ids],
        "distances": [distances],
        "metadatas": [metadatas],
        "documents": [documents],
    }
    if embeddings is not None:
        response["embeddings"] = [embeddings]
    return response
