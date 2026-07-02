"""Contract tests for extraction/utils/tokenization.py.

The tokenizer is the one place chunkers turn text into tokens; these pin the public
surface (`tokenize_document`, `token_spans`, `count_tokens`) against the injected
FakeTokenizer so the suite stays offline. Assertions target the *contract* — IDs are
ints, spans locate their tokens in the text, count matches span length — not MiniLM's
specific sub-word IDs.
"""

from tests.conftest import make_document


class TestTokenizeDocument:
    def test_returns_token_ids(self, fake_tokenizer):
        from extraction.utils.tokenization import tokenize_document

        doc = make_document(body="alpha beta gamma")
        ids = tokenize_document(doc)
        assert ids == [1, 2, 3]

    def test_empty_body_yields_no_tokens(self, fake_tokenizer):
        from extraction.utils.tokenization import tokenize_document

        assert tokenize_document(make_document(body="")) == []


class TestTokenSpans:
    def test_spans_locate_each_token_in_text(self, fake_tokenizer):
        from extraction.utils.tokenization import token_spans

        text = "alpha beta gamma"
        spans = token_spans(text)
        assert [text[s:e] for s, e in spans] == ["alpha", "beta", "gamma"]

    def test_span_count_equals_token_count(self, fake_tokenizer):
        from extraction.utils.tokenization import count_tokens, token_spans

        text = "one two three four five"
        assert len(token_spans(text)) == count_tokens(text)


class TestCountTokens:
    def test_counts_whitespace_runs(self, fake_tokenizer):
        from extraction.utils.tokenization import count_tokens

        assert count_tokens("a b c d") == 4

    def test_empty_text_is_zero(self, fake_tokenizer):
        from extraction.utils.tokenization import count_tokens

        assert count_tokens("   ") == 0


class TestLazyLoading:
    def test_get_tokenizer_returns_injected_fake_without_transformers(self, fake_tokenizer):
        # The fixture populated the cache, so _get_tokenizer must not attempt an import.
        from extraction.utils import tokenization

        assert tokenization._get_tokenizer() is fake_tokenizer
