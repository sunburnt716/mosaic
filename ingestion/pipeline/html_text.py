"""Shared HTML-to-plain-text conversion.

One definition of "strip HTML to clean text, preserving paragraph boundaries as newlines",
imported by both the normalizer (cleaning adapter `raw_body`) and body enrichment (cleaning
a fetched filing/article page). Keeping it here means the two never drift on how tags,
block boundaries, and entities are handled.
"""

import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_BLOCK_CLOSE_RE = re.compile(r"(?i)</(p|div|li|h[1-6]|section|article)>")
_BR_RE = re.compile(r"(?i)<br\s*/?>")
# <script>/<style> bodies are never content — drop them wholesale before tag-stripping so
# their inner text (JS/CSS) doesn't leak into the plain text of a fetched page.
_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1>")


def clean_html(raw_html: str | None) -> str:
    """Strip HTML to clean plain text, preserving paragraph boundaries as newlines."""
    if not raw_html:
        return ""
    text = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    text = _BLOCK_CLOSE_RE.sub("\n", text)
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
