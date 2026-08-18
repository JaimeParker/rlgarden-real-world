from __future__ import annotations

import pytest


def test_real_world_registry_rejects_dry_run_via_single_parse(monkeypatch):
    from rl_garden.training.algorithm_registry import BaseAlgorithmRegistry
    from rl_garden_real_world.training._registry import registry

    assert registry.supports_dry_run is False

    call_count = 0
    original_parse_command = BaseAlgorithmRegistry.parse_command

    def counting_parse_command(self, args=None):
        nonlocal call_count
        call_count += 1
        return original_parse_command(self, args)

    monkeypatch.setattr(BaseAlgorithmRegistry, "parse_command", counting_parse_command)

    with pytest.raises(SystemExit, match="not supported"):
        registry.run_cli(["serl", "--dry-run", "--log-type", "none"])

    assert call_count == 1
