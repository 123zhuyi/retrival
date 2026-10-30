"""PythonBuilder — importlib-based loader for Python DSL kernel solutions.

Handles: Python, Triton, TileLang (all loaded via importlib).
Sources are staged by Runner.pre_build(); build() just logs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution, SupportedLanguages
from npu_kernelbench.builder.builder import Builder


class PythonBuilder(Builder):
    """Loads Python DSL kernel solutions via importlib."""

    name = "python"

    def can_build(self, solution: Solution) -> bool:
        lang = solution.spec.languages
        if SupportedLanguages.PYTHON in lang:
            return True
        if SupportedLanguages.TILELANG in lang:
            try:
                import tilelang  # noqa: F401
                return True
            except ImportError:
                return False
        if SupportedLanguages.TRITON in lang:
            try:
                import triton  # noqa: F401
                return True
            except ImportError:
                return False
        return False

    def build(self, definition: Definition, solution: Solution,
              work_dir: str = "/tmp/npu-kernelbench",
              compile_timeout: int = 120) -> None:
        work_path = Path(work_dir)
        log = logging.getLogger(__name__)
        log.setLevel(logging.DEBUG)
        log.info("Work directory: %s", work_path)
        log.info("Python solution — no compilation needed")
