"""Custom input generator for GroupedMatmulSwigluQuantV2.

Generates int8 x/weight, wraps weight and weight_scale in Python lists, builds
group_list from the JSON value list as a concrete int64 tensor, and forces
optional tensors to None.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class GroupedMatmulSwigluQuantV2InputGenerator(BaseInputGenerator):
    """Generates inputs for the GroupedMatmulSwigluQuantV2 operator."""

    names = ("GroupedMatmulSwigluQuantV2",)
    priority = 1
    defaults = {
        "smooth_scale": None,
        "weight_assist_matrix": None,
        "bias": None,
        "dequant_mode": 0,
        "dequant_dtype": 0,
        "quant_mode": 0,
        "quant_dtype": 0,
        "group_list_type": 0,
        "tuning_config": None,
    }

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "x":
                out[name] = self._quant_tensor(ctx, "x")
            elif name == "weight":
                out[name] = [self._quant_tensor(ctx, "weight")]
            elif name == "weight_scale":
                out[name] = [ctx.rand_tensor("weight_scale")]
            elif name == "x_scale":
                out[name] = ctx.rand_tensor("x_scale")
            elif name == "group_list":
                out[name] = self._make_group_list(ctx)
            elif name in ("smooth_scale", "weight_assist_matrix", "bias"):
                out[name] = None
            elif name in self.defaults and input_spec is None:
                out[name] = self.defaults[name]
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _quant_tensor(self, ctx: InputGenerationContext, name: str) -> torch.Tensor:
        """Generate int8 tensors in the [-128, 127) range."""
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed(name))
        return torch.randint(-128, 127, list(shape), dtype=dtype, generator=gen).to(device)

    def _make_group_list(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Build group_list from the JSON value list as an int64 tensor."""
        values = ctx.scalar("group_list")
        if not isinstance(values, list):
            raise ValueError("group_list must be specified as a scalar list value")
        return torch.tensor(values, dtype=torch.int64, device=ctx.device)
