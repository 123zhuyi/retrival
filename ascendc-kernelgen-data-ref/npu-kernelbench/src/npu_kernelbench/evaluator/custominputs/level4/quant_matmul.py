"""Custom input generator for QuantMatmul.

Reproduces the old benchmark's int8 x1/x2 generation and independent
normal/uniform distribution for floating-point scales and biases.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


def _crop_float32_to_19bit(scale: torch.Tensor) -> torch.Tensor:
    """Mirror the legacy npu_trans_quant_param preprocessing."""
    if scale.dtype != torch.float32:
        return scale
    device = scale.device
    scale_cpu = scale.cpu()
    cropped = (scale_cpu.view(torch.int32) & 0xFFFFE000).view(torch.float32)
    return cropped.to(device)


class QuantMatmulInputGenerator(BaseInputGenerator):
    """Generates inputs for the QuantMatmul operator."""

    names = ("QuantMatmul",)
    priority = 1
    defaults = {
        "offset": None,
        "pertoken_scale": None,
        "bias": None,
        "output_dtype": None,
        "group_sizes": None,
    }

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name in ("x1", "x2"):
                out[name] = self._quant_tensor(ctx, name)
            elif name in ("scale", "bias", "pertoken_scale", "offset"):
                out[name] = self._optional_tensor(ctx, name)
            elif name == "output_dtype":
                out[name] = self._output_dtype(ctx)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif name in self.defaults:
                out[name] = self.defaults[name]
            else:
                out[name] = None

        scale = out.get("scale")
        if (
            isinstance(scale, torch.Tensor)
            and scale.dtype == torch.float32
            and out.get("pertoken_scale") is None
            and out.get("output_dtype") not in (torch.bfloat16, torch.int32)
        ):
            out["scale"] = _crop_float32_to_19bit(scale)
        return [out[name] for name in ctx.definition.inputs]

    def _quant_tensor(self, ctx: InputGenerationContext, name: str) -> torch.Tensor:
        """Integer tensors use randint(-5, 5); floats fall back to mixed."""
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        return ctx.rand_tensor(name, shape=shape, dtype=dtype)

    def _optional_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Generate optional tensor if present in workload, otherwise None."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        return ctx.rand_tensor(name)

    def _output_dtype(self, ctx: InputGenerationContext) -> torch.dtype | None:
        spec = ctx.workload.inputs.get("output_dtype")
        if not isinstance(spec, ScalarInput):
            return None
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int8": torch.int8,
        }.get(spec.value, spec.value)
