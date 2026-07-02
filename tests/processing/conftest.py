"""
Offline test fixtures for Phase 1 chunking.

Provides `fake_tokenizer`: a word-level stand-in installed as the cached MiniLM tokenizer so
chunker tests run without downloading a model or importing `transformers` (mirrors how the
ingestion suite fakes the network). Document builders live in tests/processing/fixtures.py.
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
    from processing.utils import tokenization

    tokenizer = FakeTokenizer()
    monkeypatch.setattr(tokenization, "_tokenizer", tokenizer)
    return tokenizer
