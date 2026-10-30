"""Custom input generator for ConvTranspose2d.

Mirrors the old benchmark: only the activation tensor ``x`` is generated;
all convolution hyper-parameters are scalar literals from the workload.
"""

from __future__ import annotations

from typing import Any

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class ConvTranspose2dInputGenerator(BaseInputGenerator):
    """Generates inputs for 2D transposed convolution."""

    names = ("ConvTranspose2d",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "x":
                out[name] = ctx.rand_tensor(name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]
