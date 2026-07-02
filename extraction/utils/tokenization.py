"""Shared tokenization — the single source of truth for turning text into tokens.

Every chunker sizes and slices by *tokens*, not characters, so retrieval-time
embedding context matches chunk-time boundaries. That tokenizer lives here and
nowhere else: chunkers import these helpers rather than each loading a tokenizer
of their own (which would risk drifting model choices — a RAG-fitness failure,
since one Chroma collection must use exactly one embedding model).

The MiniLM tokenizer is loaded lazily and cached at module level: heavy runtime
deps (`transformers`) are only imported the first time a token function is
actually called, so the offline unit suite — which monkeypatches `_tokenizer`
with a lightweight fake — never pulls them in (mirrors how the ingestion adapters
import `requests`/`feedparser` lazily inside their network methods).

Exports:
  tokenize_document(document) -> list[int]   token IDs of the document body
  token_spans(text)           -> list[(start, end)]  char offset per token
  count_tokens(text)          -> int         number of tokens in text

`token_spans` is how a token window is mapped back to a character span: it is the
`return_offsets_mapping=True` output of the fast tokenizer. `tokenize_document`
returns IDs only, per the contract — offsets are retrieved separately when needed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.core.document import Document

# One embedding model per collection — the tokenizer must match the embedder MiniLM.
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazily populated cache. None until the first real tokenization; tests overwrite it
# with a fake so the model is never downloaded offline.
_tokenizer = None


def _get_tokenizer():
    """Return the cached MiniLM fast tokenizer, loading it once on first use."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return _tokenizer


def _encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Encode text once, returning (token_ids, char_offsets) with no special tokens.

    Special tokens are excluded so every returned offset maps to real content — a
    [CLS]/[SEP] pair would otherwise contribute (0, 0) spans that slice to empty text.
    """
    tok = _get_tokenizer()
    encoded = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(encoded["input_ids"])
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return ids, offsets


def tokenize_document(document: "Document") -> list[int]:
    """Tokenize a Document's body, returning MiniLM token IDs."""
    return _encode(document.body)[0]


def token_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start_char, end_char) span of each token in text.

    The list length equals the token count; `spans[i]` locates token `i` in `text`,
    so a token window `spans[a:b]` covers characters `spans[a][0] .. spans[b-1][1]`.
    """
    return _encode(text)[1]


def count_tokens(text: str) -> int:
    """Return the number of tokens in text (used for size checks and merge decisions)."""
    return len(_encode(text)[0])
