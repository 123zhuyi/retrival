"""Registry for custom input generators."""

from __future__ import annotations

from typing import Optional

from npu_kernelbench.data.definition import Definition

from .base import BaseInputGenerator
from .default import DefaultInputGenerator


class InputGeneratorRegistry:
    """Priority-ordered registry of input generators.

    Mirrors the design of ``EvaluatorRegistry`` and ``BuilderRegistry``:
    higher-priority generators are consulted first; the built-in default
    generator always acts as the final fallback.
    """

    def __init__(self) -> None:
        self._generators: list[BaseInputGenerator] = []
        self._register_builtin(DefaultInputGenerator())

    def _register_builtin(self, generator: BaseInputGenerator) -> None:
        self._generators.append(generator)

    def register(self, generator: BaseInputGenerator) -> None:
        """Register a generator with higher priority than built-ins."""
        self._generators.insert(0, generator)

    def resolve(self, definition: Definition) -> BaseInputGenerator:
        """Return the highest-priority generator that can handle ``definition``."""
        for generator in self._generators:
            if generator.can_generate(definition):
                return generator
        raise RuntimeError(
            f"No input generator found for definition '{definition.name}'. "
            f"This should never happen because the default generator is always registered."
        )


_default_registry: Optional[InputGeneratorRegistry] = None


def get_input_generator_registry() -> InputGeneratorRegistry:
    """Return the global input generator registry, lazily initialized."""
    global _default_registry
    if _default_registry is None:
        _default_registry = InputGeneratorRegistry()
    return _default_registry
