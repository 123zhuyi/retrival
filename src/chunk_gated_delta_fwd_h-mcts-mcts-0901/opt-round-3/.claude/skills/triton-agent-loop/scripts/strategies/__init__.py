#!/usr/bin/env python3
"""Pluggable direction/baseline selection strategies."""

from .base import Strategy
from .loader import discover_strategies, load_strategy, load_strategy_for_algorithm

__all__ = ["Strategy", "discover_strategies", "load_strategy", "load_strategy_for_algorithm"]
