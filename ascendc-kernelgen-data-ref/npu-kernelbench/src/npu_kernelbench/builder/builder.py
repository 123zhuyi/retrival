"""Builder ABC — compiles a Solution into build artifacts in work_dir."""

from __future__ import annotations

from abc import ABC, abstractmethod

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.solution import Solution


class BuildError(RuntimeError):
    """Raised when a Builder fails to compile a Solution."""

    def __init__(self, message: str, builder_name: str = ""):
        self.builder_name = builder_name
        super().__init__(f"[{builder_name}] {message}" if builder_name else message)


class Builder(ABC):
    """Abstract builder that compiles a Solution into build artifacts.

    Each concrete Builder handles one language (Python or Ascend C).
    Builders are tried in priority order by BuilderRegistry. build() is a
    compile-only side effect; it returns nothing.
    """

    name: str = "base"

    @abstractmethod
    def can_build(self, solution: Solution) -> bool:
        """Return True if this builder can handle the given solution."""
        ...

    @abstractmethod
    def build(self, definition: Definition, solution: Solution,
              work_dir: str = "/tmp/npu-kernelbench",
              compile_timeout: int = 120) -> None:
        """Compile/build the solution into work_dir as a side effect.

        For Ascend C this writes kernel.so into work_dir; for Python solutions
        there is nothing to compile. Returns nothing -- the kernel function is
        loaded later by the Runner's worker via importlib.

        Raises BuildError if compilation fails.
        """
        ...
