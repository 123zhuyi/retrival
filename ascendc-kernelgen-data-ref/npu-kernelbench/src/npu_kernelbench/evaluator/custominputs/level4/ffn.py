"""Custom input generator for FFN.

Reproduces the old benchmark's dtype-aware tensor generation and optional
quantization-parameter generation.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FFNInputGenerator(BaseInputGenerator):
    """Generates inputs for the FFN operator."""

    names = ("FFN",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name in ("x", "weight1", "weight2"):
                out[name] = self._weight_like_tensor(ctx, name)
            elif name in ("bias1", "bias2"):
                out[name] = self._optional_tensor(ctx, name, is_bias=True)
            elif name in ("scale", "offset", "deq_scale1", "deq_scale2"):
                out[name] = self._optional_tensor(ctx, name)
            elif name in (
                "antiquant_scale1",
                "antiquant_scale2",
                "antiquant_offset1",
                "antiquant_offset2",
            ):
                out[name] = self._optional_tensor(ctx, name)
            elif name in ("expert_tokens", "expert_tokens_index"):
                out[name] = self._scalar_or_none(ctx, name)
            elif name == "output_dtype":
                out[name] = self._output_dtype(ctx)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _weight_like_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor:
        """Main FFN tensors: int8 weights use randint(-5,5), floats use mixed."""
        dtype = ctx.resolve_dtype(name)
        shape = ctx.resolve_shape(name)
        return ctx.rand_tensor(name, shape=shape, dtype=dtype)

    def _optional_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
        is_bias: bool = False,
    ) -> torch.Tensor | None:
        """Generate optional tensor if present in workload, otherwise None."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        dtype = ctx.resolve_dtype(name)
        shape = ctx.resolve_shape(name)
        return ctx.rand_tensor(name, shape=shape, dtype=dtype)

    def _scalar_or_none(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None

    def _output_dtype(self, ctx: InputGenerationContext) -> torch.dtype | None:
        """Convert the legacy enum value to the torch dtype passed to npu_ffn."""
        value = self._scalar_or_none(ctx, "output_dtype")
        return {
            0: torch.float16,
            1: torch.bfloat16,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(value)
