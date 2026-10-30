"""Custom input generator for GroupNormSwish.

Reproduces the old benchmark's per-case mixed distribution for the input
and fixed weight=ones / bias=zeros.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class GroupNormSwishInputGenerator(BaseInputGenerator):
    """Generates inputs for the GroupNormSwish operator."""

    names = ("GroupNormSwish",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "input":
                out[name] = self._mixed_input(ctx)
            elif name == "num_groups":
                out[name] = input_spec.value if isinstance(input_spec, ScalarInput) else 1
            elif name == "weight":
                out[name] = torch.ones(
                    ctx.resolve_shape(name), dtype=ctx.resolve_dtype(name), device=ctx.device
                )
            elif name == "bias":
                out[name] = torch.zeros(
                    ctx.resolve_shape(name), dtype=ctx.resolve_dtype(name), device=ctx.device
                )
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _mixed_input(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Use the workload's mixed distribution for the input tensor."""
        return ctx.rand_tensor("input")
