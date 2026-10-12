#!/usr/bin/env python3
"""Strategy discovery and loader.

Each supported optimization algorithm has a corresponding Strategy subclass
whose ``name`` attribute matches the algorithm name.  The Orchestrator loads
the strategy for the user-selected algorithm and passes the algorithm's
``params`` block to its constructor.
"""

import importlib
import inspect
from pathlib import Path
from typing import Any

from .base import Strategy


def _strategy_directory() -> Path:
    """Return the directory containing this package's strategy modules."""
    return Path(__file__).resolve().parent


def _load_module_safely(module_name: str) -> Any:
    """Import a module, returning None on failure."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _scan_module_for_strategies(module: Any, discovered: dict[str, type[Strategy]]) -> None:
    """Register concrete Strategy subclasses found in a module."""
    if module is None:
        return
    for _attr_name, attr in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(attr, Strategy)
            and attr is not Strategy
            and getattr(attr, "name", "")
            and attr.name not in discovered
        ):
            discovered[attr.name] = attr


def discover_strategies(strategy_dir: Path | None = None) -> dict[str, type[Strategy]]:
    """Scan strategy_dir for concrete Strategy subclasses.

    Returns a mapping from strategy/algorithm name to strategy class.
    """
    directory = strategy_dir or _strategy_directory()
    discovered: dict[str, type[Strategy]] = {}

    if not directory.is_dir():
        return discovered

    # Discover modern per-algorithm folders: strategies/{name}/strategy.py
    for subdir in sorted(directory.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        strategy_file = subdir / "strategy.py"
        if not strategy_file.is_file():
            continue
        module_name = f"strategies.{subdir.name}.strategy"
        module = _load_module_safely(module_name)
        _scan_module_for_strategies(module, discovered)

    # Discover legacy top-level modules: strategies/{name}.py
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"strategies.{path.stem}"
        module = _load_module_safely(module_name)
        _scan_module_for_strategies(module, discovered)

    return discovered


def load_strategy_for_algorithm(
    config: dict[str, Any],
    algorithm: str,
    strategy_dir: Path | None = None,
    params: dict[str, Any] | None = None,
) -> Strategy:
    """Instantiate the strategy that corresponds to *algorithm*.

    Args:
        config: Configuration dict. Must contain an ``algorithms`` block with
                the selected algorithm's ``params``.
        algorithm: Canonical algorithm name, e.g. "naive", "mcts",
                   "fusion-search".
        strategy_dir: Optional override for the directory to scan.
        params: Optional explicit strategy parameters. When provided, these
                override the parameters read from ``config["algorithms"][algorithm]``.

    Raises:
        ValueError: If the requested algorithm has no matching strategy class.
    """
    registry = discover_strategies(strategy_dir)
    if algorithm not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

    if params is None:
        algorithms = config.get("algorithms") or {}
        if not isinstance(algorithms, dict):
            algorithms = {}
        params = algorithms.get(algorithm, {}).get("params") or {}
        if not isinstance(params, dict):
            params = {}

    strategy_cls = registry[algorithm]
    try:
        return strategy_cls(**params)
    except TypeError as exc:
        raise ValueError(f"Failed to instantiate strategy '{algorithm}': {exc}") from exc


# Backward-compatible alias for callers that still pass a legacy strategy block.
# New code should use load_strategy_for_algorithm().
def load_strategy(config: dict[str, Any], strategy_dir: Path | None = None) -> Strategy:
    """Legacy loader: derive the algorithm from the legacy ``strategy`` block.

    This function is kept only for transitional callers (e.g. submit_round.py
    when the active algorithm cannot be determined).  It defaults to naive.
    """
    strategy_cfg = config.get("strategy") or {"name": "naive", "params": {}}
    name = strategy_cfg.get("name") or "naive"
    params = strategy_cfg.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    registry = discover_strategies(strategy_dir)
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")

    strategy_cls = registry[name]
    try:
        return strategy_cls(**params)
    except TypeError as exc:
        raise ValueError(f"Failed to instantiate strategy '{name}': {exc}") from exc
