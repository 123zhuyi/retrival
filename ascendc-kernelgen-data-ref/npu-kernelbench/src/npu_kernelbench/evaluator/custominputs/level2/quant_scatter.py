"""Custom input generator for QuantScatter.

Preserves the old benchmark's int8 input range [0, 127), axis-aware int32
indices, and mixed normal/uniform float distributions for updates and
quantization parameters.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class QuantScatterInputGenerator(BaseInputGenerator):
    """Generates inputs for the QuantScatter operator."""

    names = ("QuantScatter",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "input":
                out[name] = self._input_tensor(ctx)
            elif name == "indices":
                out[name] = self._indices_tensor(ctx)
            elif name == "updates":
                out[name] = self._mixed_float_tensor(ctx, name)
            elif name == "quant_scales":
                out[name] = self._positive_float_tensor(ctx, name)
            elif name == "quant_zero_points":
                out[name] = self._quant_zero_points(ctx)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _input_tensor(self, ctx: InputGenerationContext) -> torch.Tensor:
        """int8 input in [0, 127] matching the old benchmark."""
        shape = ctx.resolve_shape("input")
        dtype = ctx.resolve_dtype("input")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("input"))
        return torch.randint(0, 127, list(shape), dtype=dtype, generator=gen).to(device)

    def _indices_tensor(self, ctx: InputGenerationContext) -> torch.Tensor:
        """int32 indices in [0, input.shape[axis]) where axis is the scalar attr."""
        shape = ctx.resolve_shape("indices")
        dtype = ctx.resolve_dtype("indices")
        device = ctx.device

        axis = self._scalar_value(ctx, "axis") or 0
        input_shape = ctx.resolve_shape("input")
        ndim = len(input_shape)
        if axis < 0:
            axis = ndim + axis
        if axis < 0 or axis >= ndim:
            axis = ndim - 1
        max_val = input_shape[axis]

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("indices"))
        return torch.randint(0, max_val, list(shape), dtype=dtype, generator=gen).to(device)

    def _mixed_float_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor:
        """Legacy mixed distribution: whole case is normal or uniform."""
        return ctx.rand_tensor(name, distribution="mixed")

    def _positive_float_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
        low: float = 1e-3,
        high: float = 5.0,
    ) -> torch.Tensor:
        """Uniform positive float tensor, avoiding zero and negative values."""
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed(name))
        return (
            torch.rand(list(shape), generator=gen, dtype=torch.float32).to(device)
            * (high - low)
            + low
        ).to(dtype)

    def _quant_zero_points(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        """int32 zero points use [0, 1000); floats use mixed distribution."""
        input_spec = ctx.workload.inputs.get("quant_zero_points")
        if input_spec is None:
            return None
        shape = ctx.resolve_shape("quant_zero_points")
        dtype = ctx.resolve_dtype("quant_zero_points")
        device = ctx.device

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("quant_zero_points"))
        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return torch.randint(0, 1000, list(shape), dtype=dtype, generator=gen).to(device)
        return self._mixed_float_tensor(ctx, "quant_zero_points")

    def _scalar_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None
