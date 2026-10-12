#!/usr/bin/env python3
"""Shared configuration loader for the multi-agent optimizer.

Looks for the plugin-level ``config.json`` and returns it with a guaranteed
``algorithms`` block.  The search order is:

1. ``cwd / "config.json"`` (the directory from which the caller runs).
2. The plugin root discovered by walking up from this module.  The plugin
   root is identified by ``.claude-plugin/plugin.json`` whose ``name`` matches
   ``triton-optimizer-multi-agent`` and a sibling ``config.json``.

If no config file is found, a default config with the full algorithm preset
list is returned, so callers always have a usable configuration.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

PLUGIN_NAME = "triton-optimizer-multi-agent"

DEFAULT_ALGORITHMS: dict[str, Any] = {
    "naive": {"enabled": True, "params": {}},
    "mcts": {
        "enabled": True,
        "params": {
            "c_uct": 1.414,
            "c_pw": 2.0,
            "alpha": 0.5,
            "failure_threshold": 3,
            "reward_output_error": -2,
            "reward_compile_error": -3,
        },
    },
    "fusion-search": {
        "enabled": True,
        "params": {
            "max_routes": 3,
            "min_expected_benefit": 0.05,
            "enable_math_transforms": True,
            "route_timeout": 3600,
            "crossover_rate": 0.7,
            "mutation_rate": 0.15,
            "elite_count": 1,
            "exploration_count": 1,
        },
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "target_speedup": 5,
    "max_rounds": 10,
    "initial_worker_timeout": 7200,
    "max_worker_timeout": 28800,
    "npu_device_id": None,
    "npu_idle": {
        "aicore": 1.0,
        "aivector": 1.0,
        "npu_util": 1.0,
        "hbm": 10.0,
    },
    "algorithms": DEFAULT_ALGORITHMS,
}


def _is_plugin_root(path: Path) -> bool:
    """Return True if *path* looks like the plugin root for this agent."""
    plugin_json = path / ".claude-plugin" / "plugin.json"
    config_json = path / "config.json"
    if not plugin_json.is_file() or not config_json.is_file():
        return False
    try:
        data = json.loads(plugin_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("name") == PLUGIN_NAME


def _find_plugin_root(start: Path) -> Path | None:
    """Walk upward from *start* until the plugin root is found."""
    for candidate in [start] + list(start.parents):
        if _is_plugin_root(candidate):
            return candidate
    return None


def _normalize_algorithm_entry(entry: Any) -> dict[str, Any]:
    """Ensure an algorithm entry has the expected shape."""
    if entry is None:
        return {"enabled": False, "params": {}}
    if not isinstance(entry, dict):
        entry = {}
    return {
        "enabled": bool(entry.get("enabled", True)),
        "params": entry.get("params") if isinstance(entry.get("params"), dict) else {},
    }


def read_config(cwd: Path | None = None) -> dict[str, Any]:
    """Load the plugin configuration with a guaranteed ``algorithms`` block.

    Args:
        cwd: Directory to check first. Defaults to the current working directory.

    Returns:
        A dict containing at least ``algorithms: {<name>: {"enabled": bool,
        "params": dict}}``.
    """
    if cwd is None:
        cwd = Path.cwd()

    config_path: Path | None = cwd / "config.json"
    if not config_path.is_file():
        plugin_root = _find_plugin_root(Path(__file__).resolve().parent)
        if plugin_root is not None:
            config_path = plugin_root / "config.json"
        else:
            config_path = None

    config: dict[str, Any]
    if config_path is not None and config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}
    else:
        config = {}

    if not isinstance(config, dict):
        config = {}

    # Migrate legacy fields before dropping them. Silently discarding them
    # makes all algorithms disabled for users with old configs.
    legacy_algorithm = config.get("algorithm")
    legacy_strategy = config.get("strategy")
    migrated: list[str] = []

    if isinstance(legacy_strategy, dict):
        strategy_name = legacy_strategy.get("name") or (
            legacy_algorithm if isinstance(legacy_algorithm, str) else None
        )
        strategy_params = (
            legacy_strategy.get("params")
            if isinstance(legacy_strategy.get("params"), dict)
            else {}
        )
        if isinstance(strategy_name, str) and strategy_name:
            if not isinstance(config.get("algorithms"), dict):
                config["algorithms"] = {}
            if strategy_name not in config["algorithms"]:
                config["algorithms"][strategy_name] = {
                    "enabled": True,
                    "params": strategy_params,
                    "_migrated_from_legacy": "strategy",
                }
                migrated.append(
                    f"Migrated legacy 'strategy.name={strategy_name}' to "
                    f"algorithms.{strategy_name}.enabled=true"
                )

    if isinstance(legacy_algorithm, str) and legacy_algorithm:
        if not isinstance(config.get("algorithms"), dict):
            config["algorithms"] = {}
        if legacy_algorithm not in config["algorithms"]:
            config["algorithms"][legacy_algorithm] = {
                "enabled": True,
                "params": {},
                "_migrated_from_legacy": "algorithm",
            }
            migrated.append(
                f"Migrated legacy 'algorithm={legacy_algorithm}' to "
                f"algorithms.{legacy_algorithm}.enabled=true"
            )

    if migrated:
        warnings.warn(
            "Deprecated config fields detected and migrated:\n  - "
            + "\n  - ".join(migrated),
            UserWarning,
            stacklevel=2,
        )

    # Drop legacy fields. The active algorithm is chosen at runtime and stored
    # in the session state file, not in config.json.
    for legacy_key in ("algorithm", "strategy", "worker_type", "fusion_search"):
        config.pop(legacy_key, None)

    # Guarantee a usable algorithms block.
    if not isinstance(config.get("algorithms"), dict):
        config["algorithms"] = {}
    algorithms = config["algorithms"]

    # Normalize user-configured algorithm entries. Do NOT automatically enable
    # every known algorithm; deleting an entry from config should disable it.
    for name in list(algorithms.keys()):
        algorithms[name] = _normalize_algorithm_entry(algorithms[name])

    # For known algorithms that are NOT configured, add them as disabled presets
    # with default params so users can opt-in later without losing the defaults.
    for name, preset in DEFAULT_ALGORITHMS.items():
        if name not in algorithms:
            algorithms[name] = _normalize_algorithm_entry(
                {"enabled": False, "params": preset.get("params", {})}
            )

    # Merge missing global defaults.
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value

    return config


def get_enabled_algorithms(config: dict[str, Any]) -> list[str]:
    """Return the list of algorithm names marked as enabled."""
    algorithms = config.get("algorithms") or {}
    if not isinstance(algorithms, dict):
        return []
    return [name for name, entry in algorithms.items() if _normalize_algorithm_entry(entry)["enabled"]]


def is_algorithm_enabled(config: dict[str, Any], algorithm: str) -> bool:
    """Return whether *algorithm* is enabled in the configuration."""
    algorithms = config.get("algorithms") or {}
    if not isinstance(algorithms, dict):
        return False
    return _normalize_algorithm_entry(algorithms.get(algorithm)).get("enabled", False)


def get_algorithm_params(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    """Return the parameter block for *algorithm*.

    Returns an empty dict if the algorithm is unknown or has no params.
    """
    algorithms = config.get("algorithms") or {}
    if not isinstance(algorithms, dict):
        return {}
    return _normalize_algorithm_entry(algorithms.get(algorithm)).get("params", {})
