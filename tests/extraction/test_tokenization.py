"""
Contract tests for processing/utils/tokenization.py.

The MiniLM tokenizer is the one place Phase 1 chunkers turn text into tokens; these pin the
public surface against the injected FakeTokenizer so the suite stays offline. Assertions target
the contract (IDs are ints, spans locate their tokens), not MiniLM's specific sub-word IDs.
"""

from __future__ import annotations

from tests.extraction.fixtures import make_document


class TestTokenizeDocument:
    def test_returns_token_ids(self, fake_tokenizer):
        from extraction.utils.tokenization import tokenize_document

        assert tokenize_document(make_document(body="alpha beta gamma")) == [1, 2, 3]

    def test_empty_body_yields_no_tokens(self, fake_tokenizer):
        from extraction.utils.tokenization import tokenize_document

        assert tokenize_document(make_document(body="")) == []


class TestTokenSpans:
    def test_spans_locate_each_token_in_text(self, fake_tokenizer):
        from extraction.utils.tokenization import token_spans

        text = "alpha beta gamma"
        spans = token_spans(text)
        assert [text[s:e] for s, e in spans] == ["alpha", "beta", "gamma"]

    def test_empty_text_has_no_spans(self, fake_tokenizer):
        from extraction.utils.tokenization import token_spans

        assert token_spans("   ") == []


class TestLazyLoading:
    def test_get_tokenizer_returns_injected_fake_without_transformers(self, fake_tokenizer):
        # The fixture populated the cache, so _get_tokenizer must not attempt an import.
        from extraction.utils import tokenization

        assert tokenization._get_tokenizer() is fake_tokenizer
