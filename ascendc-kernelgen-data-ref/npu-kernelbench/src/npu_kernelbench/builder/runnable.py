"""Runnable — uniform wrapper around any built/loaded kernel function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Runnable:
    """Uniform callable wrapper for any compiled kernel.

    Provides a single interface regardless of source language (Python,
    Ascend C, Triton, Torch extension).
    """
    fn: Callable[..., Any]
    definition_name: str
    solution_name: str
    destination_passing_style: bool = False
    _cleanup: Optional[Callable[[], None]] = field(default=None, repr=False)

    def __call__(self, *args: Any) -> Any:
        """Positional call (primary interface)."""
        return self.fn(*args)

    def close(self) -> None:
        """Release resources (e.g., dlclose shared library)."""
        if self._cleanup:
            self._cleanup()
