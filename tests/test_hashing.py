"""Tests for ingestion/pipeline/hashing.py.

Contract under test:
  content_hash(raw_body: str | bytes) -> str
    SHA-256 hex of whitespace-normalized content. Always 64 chars, lowercase, no dashes.

  identity_key(source_name: str, source_article_id: str) -> str
    Stable key for the same logical article across updates: "{source_name}::{source_article_id}"

  document_id(identity_key: str, content_hash: str) -> str
    Globally unique Document.id derived from both inputs. Changes when content changes.
    64-char hex, lowercase, no dashes — safe as a Chroma document ID.
"""

from ingestion.pipeline.hashing import content_hash, document_id, identity_key


class TestContentHash:
    def test_returns_64_char_hex_string(self):
        result = content_hash("Some article text about the Fed")
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_input_same_hash(self):
        text = "The Fed cut rates by 25bps."
        assert content_hash(text) == content_hash(text)

    def test_different_content_different_hash(self):
        assert content_hash("article one: Fed raises rates") != content_hash(
            "article two: Fed cuts rates"
        )

    def test_leading_trailing_whitespace_normalized(self):
        assert content_hash("Fed cuts rates") == content_hash("  Fed cuts rates  ")

    def test_internal_whitespace_runs_normalized(self):
        assert content_hash("Fed  cuts  rates") == content_hash("Fed cuts rates")

    def test_newlines_treated_as_whitespace(self):
        assert content_hash("Fed cuts\nrates") == content_hash("Fed cuts rates")

    def test_accepts_bytes_input(self):
        result = content_hash(b"raw bytes content from the wire")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_str_and_utf8_bytes_produce_same_hash(self):
        text = "Fed rate cut announced Wednesday"
        assert content_hash(text) == content_hash(text.encode("utf-8"))

    def test_lowercase_output(self):
        result = content_hash("any content")
        assert result == result.lower()

    def test_no_dashes_in_output(self):
        assert "-" not in content_hash("any content")


class TestIdentityKey:
    def test_format_is_source_double_colon_article_id(self):
        key = identity_key("Reuters", "newsml_L1N3D30GH")
        assert key == "Reuters::newsml_L1N3D30GH"

    def test_deterministic(self):
        assert identity_key("Reuters", "newsml_L1N3D30GH") == identity_key(
            "Reuters", "newsml_L1N3D30GH"
        )

    def test_different_sources_produce_different_keys(self):
        assert identity_key("Reuters", "article-1") != identity_key(
            "Bloomberg", "article-1"
        )

    def test_different_article_ids_produce_different_keys(self):
        assert identity_key("Reuters", "article-1") != identity_key(
            "Reuters", "article-2"
        )

    def test_edgar_accession_number_as_article_id(self):
        key = identity_key("SEC EDGAR", "0001193125-24-002345")
        assert key == "SEC EDGAR::0001193125-24-002345"

    def test_rss_guid_as_article_id(self):
        key = identity_key("Reuters", "tag:reuters.com,2024:newsml_L1N3D30GH")
        assert "::" in key
        assert key.startswith("Reuters::")


class TestDocumentId:
    def test_returns_64_char_hex_string(self):
        result = document_id("Reuters::newsml_L1N3D30GH", "a" * 64)
        assert isinstance(result, str)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        ikey = "Reuters::newsml_L1N3D30GH"
        chash = "a" * 64
        assert document_id(ikey, chash) == document_id(ikey, chash)

    def test_changes_when_content_hash_changes(self):
        ikey = "Reuters::newsml_L1N3D30GH"
        id_v1 = document_id(ikey, "a" * 64)
        id_v2 = document_id(ikey, "b" * 64)
        assert id_v1 != id_v2

    def test_changes_when_identity_key_changes(self):
        chash = "a" * 64
        id_a1 = document_id("Reuters::article-1", chash)
        id_a2 = document_id("Reuters::article-2", chash)
        assert id_a1 != id_a2

    def test_lowercase_output(self):
        result = document_id("Reuters::article-1", "a" * 64)
        assert result == result.lower()

    def test_no_dashes_in_output(self):
        assert "-" not in document_id("Reuters::article-1", "a" * 64)

    def test_same_content_different_sources_produce_different_ids(self):
        chash = "a" * 64
        reuters_id = document_id("Reuters::article-1", chash)
        bloomberg_id = document_id("Bloomberg::article-1", chash)
        assert reuters_id != bloomberg_id
