"""BuilderRegistry — priority-ordered builders with content-addressable cache."""

from __future__ import annotations

from typing import Optional

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution
from npu_kernelbench.builder.builder import Builder, BuildError
from npu_kernelbench.builder.python_builder import PythonBuilder
from npu_kernelbench.builder.ascendc_builder import AscendCBuilder


class BuilderRegistry:
    """Singleton registry with priority-ordered builders and SHA256 cache."""

    def __init__(self):
        self._builders: list[Builder] = [
            PythonBuilder(),     # priority 1 — Python DSL (PyTorch, Triton, TileLang)
            AscendCBuilder(),    # priority 2 — NPU native Ascend C
        ]

    def register(self, builder: Builder, prepend: bool = True) -> None:
        """Register a builder. prepend=True puts it first (higher priority)."""
        if prepend:
            self._builders.insert(0, builder)
        else:
            self._builders.append(builder)

    def build(self, definition: Definition, solution: Solution,
              work_dir: str = "/tmp/npu-kernelbench",
              compile_timeout: int = 120) -> None:
        """Build a solution. Raises BuildError if all builders fail."""
        last_error: Optional[BuildError] = None
        for builder in self._builders:
            if not builder.can_build(solution):
                continue
            try:
                builder.build(definition, solution, work_dir=work_dir,
                              compile_timeout=compile_timeout)
                return
            except BuildError as e:
                last_error = e
                continue

        raise BuildError(
            f"No builder succeeded for solution '{solution.name}'. "
            f"Last error: {last_error}"
        )


_default_registry: Optional[BuilderRegistry] = None


def get_registry() -> BuilderRegistry:
    """Get or create the global BuilderRegistry singleton."""
    global _default_registry
    if _default_registry is None:
        _default_registry = BuilderRegistry()
    return _default_registry
