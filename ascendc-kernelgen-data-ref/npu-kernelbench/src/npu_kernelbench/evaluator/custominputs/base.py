"""Base class for custom input generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from npu_kernelbench.data.definition import Definition
from npu_kernelbench.data.workload import (
    CustomInput,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    TensorListInput,
)

from .context import InputGenerationContext


class BaseInputGenerator(ABC):
    """Abstract base class for per-operator input generation.

    Subclasses declare the operator names they handle and implement
    ``generate()`` to produce input values in ``definition.inputs`` order.
    """

    names: tuple[str, ...] = ()
    """Operator names this generator handles. Empty tuple means it never
    matches via ``can_generate`` (useful for base/mixin classes)."""

    priority: int = 0
    """Higher priority generators are consulted first during resolution."""

    def can_generate(self, definition: Definition) -> bool:
        """Return True if this generator handles the given definition."""
        return definition.name in self.names

    @abstractmethod
    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        """Generate inputs for ``ctx.definition`` in input-name order.

        Args:
            ctx: Generation context with RNG state, device, helpers, etc.

        Returns:
            List of input values (tensors, scalars, lists, None) matching
            ``definition.inputs`` order.
        """
        ...

    def _get_input_value(self, ctx: InputGenerationContext, name: str) -> Any:
        """Return the input value for ``name``, falling back to reference defaults.

        If the workload provides the input, use it. Otherwise, if the reference
        ``run`` function declares a default for this parameter (e.g. ``scale=1.0``),
        return that default instead of ``None``. This mirrors the legacy framework
        behaviour for optional scalar parameters.
        """
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is not None:
            if isinstance(input_spec, ScalarInput):
                # Explicit null scalar in the workload still means "use default".
                if input_spec.value is not None:
                    return input_spec.value
            elif isinstance(input_spec, RandomInput):
                return ctx.rand_tensor(name)
            elif isinstance(input_spec, TensorListInput):
                return ctx.rand_tensor(name)
            elif isinstance(input_spec, CustomInput):
                return ctx.custom_tensor(name)
            elif isinstance(input_spec, SafetensorsInput):
                return ctx.safetensor(name)
        return ctx.definition.input_defaults.get(name)
