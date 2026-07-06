"""
Tests for ingestion/core/source_config.py.

Covers:
  - _parse_interval: valid formats, invalid formats, zero duration.
  - _validate_entry: each required field missing or of the wrong type.
  - load_sources: happy path, file not found, bad YAML structure,
    duplicate names, all required and optional fields.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from ingestion.core.source_config import SourceConfig, _parse_interval, load_sources

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(tmp_path: Path, data: dict) -> Path:
    """Write a YAML file to tmp_path and return its path."""
    p = tmp_path / "sources.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def minimal_entry(**overrides) -> dict:
    """Return a minimal valid source entry with optional field overrides."""
    base = {
        "name": "test-source",
        "adapter": "rss",
        "tier": 1,
        "url": "https://example.com/feed.xml",
        "poll_interval": "5m",
    }
    base.update(overrides)
    return base


def minimal_config(*entries) -> dict:
    """Wrap entries in the top-level YAML structure."""
    return {"sources": list(entries) or [minimal_entry()]}


# ---------------------------------------------------------------------------
# _parse_interval
# ---------------------------------------------------------------------------


class TestParseInterval:
    def test_minutes_only(self):
        assert _parse_interval("5m") == timedelta(minutes=5)

    def test_hours_only(self):
        assert _parse_interval("1h") == timedelta(hours=1)

    def test_seconds_only(self):
        assert _parse_interval("30s") == timedelta(seconds=30)

    def test_hours_and_minutes(self):
        assert _parse_interval("1h30m") == timedelta(hours=1, minutes=30)

    def test_hours_minutes_seconds(self):
        assert _parse_interval("2h15m30s") == timedelta(hours=2, minutes=15, seconds=30)

    def test_large_minutes_normalised(self):
        # 90m == 1h30m; timedelta normalises this internally
        assert _parse_interval("90m") == timedelta(minutes=90)

    def test_whitespace_stripped(self):
        assert _parse_interval("  5m  ") == timedelta(minutes=5)

    def test_zero_seconds_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _parse_interval("0s")

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _parse_interval("0h0m0s")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid poll_interval"):
            _parse_interval("")

    def test_plain_number_raises(self):
        with pytest.raises(ValueError, match="Invalid poll_interval"):
            _parse_interval("300")

    def test_natural_language_raises(self):
        with pytest.raises(ValueError, match="Invalid poll_interval"):
            _parse_interval("5 minutes")

    def test_uppercase_raises(self):
        # Only lowercase h/m/s are accepted (spec uses lowercase examples).
        with pytest.raises(ValueError, match="Invalid poll_interval"):
            _parse_interval("5M")

    def test_float_raises(self):
        with pytest.raises(ValueError, match="Invalid poll_interval"):
            _parse_interval("1.5h")

    def test_negative_raises(self):
        # Regex doesn't match a leading "-"; treated as invalid format.
        with pytest.raises(ValueError, match="Invalid poll_interval"):
            _parse_interval("-5m")


# ---------------------------------------------------------------------------
# load_sources — file-level errors
# ---------------------------------------------------------------------------


class TestLoadSourcesFileErrors:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_sources(tmp_path / "nonexistent.yaml")

    def test_empty_sources_list_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, {"sources": []})
        with pytest.raises(ValueError, match="non-empty list"):
            load_sources(p)

    def test_missing_sources_key_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, {"feeds": [minimal_entry()]})
        with pytest.raises(ValueError, match="'sources' key"):
            load_sources(p)

    def test_sources_not_a_list_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, {"sources": "not-a-list"})
        with pytest.raises(ValueError):
            load_sources(p)

    def test_entry_not_a_mapping_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, {"sources": ["just-a-string"]})
        with pytest.raises(ValueError, match="not a mapping"):
            load_sources(p)


# ---------------------------------------------------------------------------
# load_sources — required field validation
# ---------------------------------------------------------------------------


class TestLoadSourcesRequiredFields:
    @pytest.mark.parametrize("missing_key", ["name", "adapter", "tier", "url", "poll_interval"])
    def test_missing_required_field_raises(self, tmp_path: Path, missing_key: str):
        entry = minimal_entry()
        del entry[missing_key]
        p = write_yaml(tmp_path, minimal_config(entry))
        with pytest.raises(ValueError, match=missing_key):
            load_sources(p)

    def test_empty_name_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(name="   ")))
        with pytest.raises(ValueError, match="non-empty string"):
            load_sources(p)

    def test_empty_url_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(url="")))
        with pytest.raises(ValueError, match="non-empty string"):
            load_sources(p)

    def test_invalid_adapter_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(adapter="kafka")))
        with pytest.raises(ValueError, match="not registered"):
            load_sources(p)

    def test_tier_out_of_range_high_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(tier=4)))
        with pytest.raises(ValueError, match="0–3"):
            load_sources(p)

    def test_tier_out_of_range_low_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(tier=-1)))
        with pytest.raises(ValueError, match="0–3"):
            load_sources(p)

    def test_tier_wrong_type_string_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(tier="high")))
        with pytest.raises(ValueError, match="integer"):
            load_sources(p)

    def test_tier_bool_rejected(self, tmp_path: Path):
        # In YAML, `true` parses to Python True (bool), which is a subclass of int.
        # We must reject it explicitly.
        p = write_yaml(tmp_path, minimal_config(minimal_entry(tier=True)))
        with pytest.raises(ValueError, match="integer"):
            load_sources(p)

    def test_poll_interval_not_string_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(poll_interval=300)))
        with pytest.raises(ValueError, match="string"):
            load_sources(p)

    def test_poll_interval_bad_format_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(poll_interval="5 min")))
        with pytest.raises(ValueError):
            load_sources(p)

    def test_poll_interval_zero_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(poll_interval="0s")))
        with pytest.raises(ValueError, match="positive"):
            load_sources(p)


# ---------------------------------------------------------------------------
# load_sources — optional field validation
# ---------------------------------------------------------------------------


class TestLoadSourcesOptionalFields:
    def test_doc_type_filing_accepted(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(doc_type="filing")))
        sources = load_sources(p)
        assert sources[0].doc_type == "filing"

    def test_doc_type_invalid_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(doc_type="post")))
        with pytest.raises(ValueError, match="doc_type"):
            load_sources(p)

    def test_doc_type_defaults_to_article(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry()))
        sources = load_sources(p)
        assert sources[0].doc_type == "article"

    def test_enabled_false_source_still_loaded(self, tmp_path: Path):
        # Filtering disabled sources is run.py's job, not load_sources.
        p = write_yaml(tmp_path, minimal_config(minimal_entry(enabled=False)))
        sources = load_sources(p)
        assert len(sources) == 1
        assert sources[0].enabled is False

    def test_enabled_non_bool_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(enabled="yes")))
        with pytest.raises(ValueError, match="boolean"):
            load_sources(p)

    def test_params_not_mapping_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(params="fast")))
        with pytest.raises(ValueError, match="mapping"):
            load_sources(p)

    def test_headers_not_str_str_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(headers={"x-count": 1})))
        with pytest.raises(ValueError, match="str→str"):
            load_sources(p)

    def test_field_mappings_not_str_str_raises(self, tmp_path: Path):
        p = write_yaml(
            tmp_path,
            minimal_config(minimal_entry(field_mappings={"body": 42})),
        )
        with pytest.raises(ValueError, match="str→str"):
            load_sources(p)

    def test_auth_not_mapping_raises(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(auth="token123")))
        with pytest.raises(ValueError, match="mapping"):
            load_sources(p)


# ---------------------------------------------------------------------------
# load_sources — uniqueness and multi-source
# ---------------------------------------------------------------------------


class TestLoadSourcesUniqueness:
    def test_duplicate_names_raise(self, tmp_path: Path):
        p = write_yaml(
            tmp_path,
            minimal_config(
                minimal_entry(name="dup"),
                minimal_entry(name="dup"),
            ),
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_sources(p)

    def test_multiple_valid_sources_all_loaded(self, tmp_path: Path):
        p = write_yaml(
            tmp_path,
            minimal_config(
                minimal_entry(name="source-a", adapter="rss"),
                minimal_entry(name="source-b", adapter="rest_json"),
                minimal_entry(name="source-c", adapter="rss", doc_type="filing"),
            ),
        )
        sources = load_sources(p)
        assert len(sources) == 3
        assert [s.name for s in sources] == ["source-a", "source-b", "source-c"]

    def test_all_valid_adapters_accepted(self, tmp_path: Path):
        # "edgar" is deliberately not a valid adapter: a specialized EDGAR adapter was
        # tried and retired (see adapters/edgar.py); EDGAR discovery runs through the
        # generic "rss" adapter plus a transform instead.
        entries = [
            minimal_entry(name="a", adapter="rss"),
            minimal_entry(name="b", adapter="rest_json"),
        ]
        p = write_yaml(tmp_path, {"sources": entries})
        sources = load_sources(p)
        adapters = {s.adapter for s in sources}
        assert adapters == {"rss", "rest_json"}


# ---------------------------------------------------------------------------
# load_sources — happy path / field values
# ---------------------------------------------------------------------------


class TestLoadSourcesHappyPath:
    def test_returns_source_config_instances(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config())
        sources = load_sources(p)
        assert all(isinstance(s, SourceConfig) for s in sources)

    def test_poll_interval_parsed_correctly(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(poll_interval="1h30m")))
        sources = load_sources(p)
        assert sources[0].poll_interval == timedelta(hours=1, minutes=30)

    def test_tier_zero_accepted(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(tier=0)))
        sources = load_sources(p)
        assert sources[0].tier == 0

    def test_tier_three_accepted(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(tier=3)))
        sources = load_sources(p)
        assert sources[0].tier == 3

    def test_field_mappings_stored(self, tmp_path: Path):
        mappings = {"body": "summary", "published_date": "published"}
        p = write_yaml(tmp_path, minimal_config(minimal_entry(field_mappings=mappings)))
        sources = load_sources(p)
        assert sources[0].field_mappings == mappings

    def test_auth_stored(self, tmp_path: Path):
        auth = {"token": "Bearer abc123"}
        p = write_yaml(tmp_path, minimal_config(minimal_entry(auth=auth)))
        sources = load_sources(p)
        assert sources[0].auth == auth

    def test_name_whitespace_stripped(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config(minimal_entry(name="  test-source  ")))
        sources = load_sources(p)
        assert sources[0].name == "test-source"

    def test_source_config_is_frozen(self, tmp_path: Path):
        p = write_yaml(tmp_path, minimal_config())
        sources = load_sources(p)
        with pytest.raises((AttributeError, TypeError)):
            sources[0].name = "mutated"  # type: ignore[misc]
