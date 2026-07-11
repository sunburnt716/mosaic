"""
Offline test fixtures for Phase 1 chunking and retrieval's query embedding.

Provides `fake_tokenizer`: a word-level stand-in installed as the cached MiniLM tokenizer so
chunker tests run without downloading a model or importing `transformers` (mirrors how the
ingestion suite fakes the network). `fake_embedder` does the same for the MiniLM embedder
(`extraction.utils.embedding`), for tests that don't need real semantic vectors. Document
builders live in tests/extraction/fixtures.py.
"""

from __future__ import annotations

import re

import pytest


class FakeTokenizer:
    """Word-level stand-in for the MiniLM fast tokenizer — offline, no model download.

    Splits on whitespace; each run of non-space characters is one token whose char offsets are
    its (start, end) in the text. Coarser than MiniLM's sub-word tokenizer, but exact and
    deterministic — enough to drive the chunkers' boundary/offset logic. Tests assert the
    contract (spans locate text, ordinals contiguous, dual-span rules), not specific token IDs.
    """

    _WORD_RE = re.compile(r"\S+")

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for i, match in enumerate(self._WORD_RE.finditer(text)):
            ids.append(i + 1)
            offsets.append((match.start(), match.end()))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


@pytest.fixture
def fake_tokenizer(monkeypatch):
    """Install the word-level FakeTokenizer as tokenization's cached tokenizer.

    Overwrites the lazily-cached `_tokenizer` global so `_get_tokenizer()` never triggers the
    real MiniLM download. Auto-restored to None after the test by monkeypatch.
    """
    from extraction.utils import tokenization

    tokenizer = FakeTokenizer()
    monkeypatch.setattr(tokenization, "_tokenizer", tokenizer)
    return tokenizer


class FakeEmbeddingModel:
    """Deterministic stand-in for the MiniLM SentenceTransformer — offline, no model download.

    Encodes a string as a fixed-length vector derived from its character codes, so equal text
    always yields equal vectors and different text yields different ones. Not semantically
    meaningful — tests assert the contract (fixed length, determinism), not real similarity.
    """

    _DIM = 8

    def encode(self, text: str) -> "_FakeVector":
        if not text:
            return _FakeVector([0.0] * self._DIM)
        return _FakeVector(
            [(sum(ord(c) for c in text[i :: self._DIM]) % 97) / 97 for i in range(self._DIM)]
        )


class _FakeVector(list):
    """list subclass that also exposes .tolist(), mirroring a real numpy array's shape.

    embed_text() calls .tolist() (not list(...)) on the model's output — a plain list
    fake would mask a regression back to list(...), which silently breaks against a real
    model (numpy scalars aren't valid Chroma embedding values). Subclassing list keeps
    every other test's iteration/equality/len assumptions unchanged.
    """

    def tolist(self) -> list[float]:
        return list(self)


@pytest.fixture
def fake_embedder(monkeypatch):
    """Install the deterministic FakeEmbeddingModel as embedding's cached model.

    Overwrites the lazily-cached `_model` global so `_get_model()` never triggers the real
    MiniLM download. Auto-restored to None after the test by monkeypatch.
    """
    from extraction.utils import embedding

    model = FakeEmbeddingModel()
    monkeypatch.setattr(embedding, "_model", model)
    return model
