"""Custom input generator for WeightQuantBatchmatmul.

Generates int8 weight in the narrow [-8, 8) range, produces optional tensors
as None when their shapes are absent, and defaults antiquant_scale to ones([1])
when it is missing from the workload.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class WeightQuantBatchmatmulInputGenerator(BaseInputGenerator):
    """Generates inputs for the WeightQuantBatchmatmul operator."""

    names = ("WeightQuantBatchmatmul",)
    priority = 1
    defaults = {
        "antiquant_offset": None,
        "quant_scale": None,
        "quant_offset": None,
        "bias": None,
        "antiquant_group_size": 0,
        "inner_precise": 0,
    }

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "x":
                out[name] = ctx.rand_tensor(name)
            elif name == "weight":
                out[name] = self._make_weight(ctx)
            elif name == "antiquant_scale":
                out[name] = self._make_antiquant_scale(ctx)
            elif name in ("antiquant_offset", "quant_scale", "quant_offset", "bias"):
                out[name] = self._optional_tensor(ctx, name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif name in self.defaults:
                out[name] = self.defaults[name]
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_weight(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Generate int8 weight in the narrow [-8, 8) range."""
        shape = ctx.resolve_shape("weight")
        dtype = ctx.resolve_dtype("weight")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("weight"))
        return torch.randint(-8, 8, list(shape), dtype=dtype, generator=gen).to(device)

    def _make_antiquant_scale(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Generate antiquant_scale or default to ones([1])."""
        input_spec = ctx.workload.inputs.get("antiquant_scale")
        if input_spec is None:
            x_dtype = ctx.resolve_dtype("x")
            return torch.ones([1], dtype=x_dtype, device=ctx.device)
        return ctx.rand_tensor("antiquant_scale")

    def _optional_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Return None if the optional tensor is absent from the workload."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        return ctx.rand_tensor(name)
