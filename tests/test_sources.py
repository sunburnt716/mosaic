"""Tests for ingestion/sources.py — the load_sources() registry loader.

Fully offline: loader mechanics run against crafted temp JSON files, and the real
config/sources.json is asserted for the invariants the spec requires (no dead Reuters,
EDGAR present, at least one Tier 1 source). No network.
"""

import json

from ingestion.core.source_config import SourceConfig
from ingestion.sources import DEFAULT_REGISTRY_PATH, load_sources


# ---------------------------------------------------------------------------
# Loader mechanics (crafted JSON via tmp_path)
# ---------------------------------------------------------------------------


class TestLoadSourcesMechanics:
    def _write(self, tmp_path, payload) -> str:
        p = tmp_path / "sources.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return str(p)

    def test_returns_typed_source_configs(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "sources": [
                    {"name": "s1", "adapter": "rss", "tier": 1, "url": "https://x/1"}
                ]
            },
        )
        configs = load_sources(path)
        assert len(configs) == 1
        assert isinstance(configs[0], SourceConfig)
        assert configs[0].name == "s1"

    def test_round_trips_transform_and_expects(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "sources": [
                    {
                        "name": "edgar",
                        "adapter": "rss",
                        "tier": 0,
                        "url": "https://x/edgar",
                        "transform": "edgar_filing_url",
                        "expects": {"title": True, "url": True, "body": False},
                    }
                ]
            },
        )
        cfg = load_sources(path)[0]
        assert cfg.transform == "edgar_filing_url"
        assert cfg.expects == {"title": True, "url": True, "body": False}

    def test_ignores_underscore_prefixed_annotation_keys(self, tmp_path):
        """Documentation-only keys like _note / _comment must not break construction."""
        path = self._write(
            tmp_path,
            {
                "_comment": "registry-level note",
                "sources": [
                    {
                        "name": "s1",
                        "adapter": "rss",
                        "tier": 1,
                        "url": "https://x/1",
                        "_note": "per-entry note",
                    }
                ],
            },
        )
        cfg = load_sources(path)[0]
        assert cfg.name == "s1"

    def test_accepts_bare_list_envelope(self, tmp_path):
        path = self._write(
            tmp_path,
            [{"name": "s1", "adapter": "rss", "tier": 1, "url": "https://x/1"}],
        )
        configs = load_sources(path)
        assert configs[0].name == "s1"

    def test_defaults_applied_for_optional_fields(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "sources": [
                    {"name": "s1", "adapter": "rss", "tier": 1, "url": "https://x/1"}
                ]
            },
        )
        cfg = load_sources(path)[0]
        assert cfg.enabled is True
        assert cfg.poll_interval == "10m"  # _DEFAULT_INTERVAL
        assert cfg.transform is None
        assert cfg.expects == {}


# ---------------------------------------------------------------------------
# The real registry (config/sources.json) — spec invariants
# ---------------------------------------------------------------------------


class TestRealRegistry:
    def test_real_registry_loads(self):
        configs = load_sources(DEFAULT_REGISTRY_PATH)
        assert all(isinstance(c, SourceConfig) for c in configs)
        assert len(configs) >= 2

    def test_no_dead_reuters_entry(self):
        """Reuters host was decommissioned June 2020; it must not be in the registry."""
        names = {c.name for c in load_sources(DEFAULT_REGISTRY_PATH)}
        assert "reuters-rss" not in names

    def test_edgar_present_and_tier0(self):
        configs = load_sources(DEFAULT_REGISTRY_PATH)
        edgar = next((c for c in configs if c.name == "sec-edgar"), None)
        assert edgar is not None
        assert edgar.tier == 0
        assert edgar.transform == "edgar_filing_url"

    def test_at_least_one_tier1_source(self):
        configs = load_sources(DEFAULT_REGISTRY_PATH)
        assert any(c.tier == 1 for c in configs)

    def test_every_source_uses_a_registered_adapter_key(self):
        # Adapters are config-selected; the registry must only name known keys.
        from ingestion.adapters.registry import _REGISTRY

        for c in load_sources(DEFAULT_REGISTRY_PATH):
            assert c.adapter in _REGISTRY, (
                f"{c.name} uses unknown adapter {c.adapter!r}"
            )
