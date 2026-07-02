"""Tests for ingestion/core/source_config.py — the SourceConfig dataclass."""

from ingestion.core.source_config import SourceConfig


def _minimal(**overrides) -> SourceConfig:
    base = dict(name="test-source", adapter="rss", tier=1, url="https://example.com")
    base.update(overrides)
    return SourceConfig(**base)


class TestSourceConfigRequiredFields:
    def test_name_stored(self):
        assert _minimal().name == "test-source"

    def test_adapter_stored(self):
        assert _minimal().adapter == "rss"

    def test_tier_stored(self):
        assert _minimal().tier == 1

    def test_url_stored(self):
        assert _minimal().url == "https://example.com"


class TestSourceConfigDefaults:
    def test_enabled_defaults_to_true(self):
        assert _minimal().enabled is True

    def test_params_defaults_to_empty_dict(self):
        assert _minimal().params == {}

    def test_headers_defaults_to_empty_dict(self):
        assert _minimal().headers == {}

    def test_poll_interval_defaults_to_none(self):
        assert _minimal().poll_interval is None

    def test_transform_defaults_to_none(self):
        assert _minimal().transform is None

    def test_expects_defaults_to_empty_dict(self):
        assert _minimal().expects == {}

    def test_quality_thresholds_default_to_none(self):
        cfg = _minimal()
        assert cfg.max_fallback_title_rate is None
        assert cfg.max_empty_body_rate is None
        assert cfg.min_records is None


class TestSourceConfigOptionalFields:
    def test_enabled_false(self):
        assert _minimal(enabled=False).enabled is False

    def test_poll_interval_stored_as_string(self):
        assert _minimal(poll_interval="5m").poll_interval == "5m"

    def test_transform_stored(self):
        assert _minimal(transform="edgar_filing_url").transform == "edgar_filing_url"

    def test_params_stored(self):
        assert _minimal(params={"doc_type": "filing"}).params == {"doc_type": "filing"}

    def test_headers_stored(self):
        assert _minimal(headers={"X-Api-Key": "abc"}).headers == {"X-Api-Key": "abc"}

    def test_expects_stored(self):
        cfg = _minimal(expects={"title": True, "url": True, "body": False})
        assert cfg.expects == {"title": True, "url": True, "body": False}

    def test_quality_thresholds_stored(self):
        cfg = _minimal(max_fallback_title_rate=0.5, max_empty_body_rate=0.2, min_records=5)
        assert cfg.max_fallback_title_rate == 0.5
        assert cfg.max_empty_body_rate == 0.2
        assert cfg.min_records == 5


class TestSourceConfigMutableDefaults:
    def test_params_instances_are_independent(self):
        a = _minimal(name="a")
        b = _minimal(name="b")
        a.params["key"] = "val"
        assert b.params == {}

    def test_headers_instances_are_independent(self):
        a = _minimal(name="a")
        b = _minimal(name="b")
        a.headers["X-Foo"] = "bar"
        assert b.headers == {}

    def test_expects_instances_are_independent(self):
        a = _minimal(name="a")
        b = _minimal(name="b")
        a.expects["title"] = True
        assert b.expects == {}
