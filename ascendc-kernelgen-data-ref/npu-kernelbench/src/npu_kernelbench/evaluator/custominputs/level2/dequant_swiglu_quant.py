"""Custom input generator for DequantSwigluQuant.

Reproduces the old benchmark's per-case distribution alternation and
post-processing of weight_scale/activation_scale/quant_scale based on x shape.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class DequantSwigluQuantInputGenerator(BaseInputGenerator):
    """Generates inputs for the DequantSwigluQuant operator."""

    names = ("DequantSwigluQuant",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        # Generate base random/scalar inputs
        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "x":
                out[name] = self._int_tensor(ctx, name, low=0, high=1000)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif name in ("weight_scale", "activation_scale", "bias", "quant_scale", "quant_offset", "group_index"):
                # placeholder; post-processed below
                out[name] = ctx.rand_tensor(name) if isinstance(input_spec, RandomInput) else None
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        # Post-process dequant/quant params exactly like the old benchmark
        x = out["x"]
        tokens_num = x.shape[0]
        hidden_size = x.shape[1]
        half_h = hidden_size // 2

        quant_mode = self._scalar_value(ctx, "quant_mode") or 0
        has_group = "group_index" in ctx.workload.inputs and ctx.workload.inputs["group_index"] is not None

        if has_group:
            quant_mode = 1
            out["bias"] = None
            out["quant_offset"] = None
            gi_shape = ctx.resolve_shape("group_index")
            num_groups = gi_shape[0]
            out["group_index"] = torch.tensor(
                [tokens_num // num_groups] * num_groups, dtype=torch.int64, device=ctx.device
            )
        else:
            if x.dtype == torch.int32:
                if "weight_scale" in ctx.workload.inputs:
                    out["weight_scale"] = ctx.rand_tensor("weight_scale")
                else:
                    out["weight_scale"] = None
                if "activation_scale" in ctx.workload.inputs:
                    out["activation_scale"] = ctx.rand_tensor("activation_scale")
                else:
                    out["activation_scale"] = None
            else:
                out["weight_scale"] = None
                out["activation_scale"] = None
                out["bias"] = None

        if "quant_scale" in ctx.workload.inputs and ctx.workload.inputs["quant_scale"] is not None:
            out["quant_scale"] = ctx.rand_tensor("quant_scale")
        else:
            out["quant_scale"] = None

        if has_group:
            out["quant_offset"] = None
        elif "quant_offset" in ctx.workload.inputs and ctx.workload.inputs["quant_offset"] is not None and quant_mode == 0:
            qs = out.get("quant_scale")
            if qs is not None:
                out["quant_offset"] = ctx.rand_tensor("quant_offset")
            else:
                out["quant_offset"] = None
        else:
            out["quant_offset"] = None

        return [out[name] for name in ctx.definition.inputs]

    def _int_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
        low: int,
        high: int,
    ) -> torch.Tensor:
        """Integer tensor matching the old benchmark's randint ranges."""
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed(name))
        return torch.randint(low, high, list(shape), dtype=dtype, generator=gen).to(ctx.device)

    def _scalar_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None
