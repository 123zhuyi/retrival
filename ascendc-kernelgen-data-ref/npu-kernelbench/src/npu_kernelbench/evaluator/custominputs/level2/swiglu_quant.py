"""Custom input generator for SwigluQuant.

Reproduces the old benchmark's mixed x generation, smooth_scales/offsets
regeneration, and group_index reconstruction based on group_list_type.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class SwigluQuantInputGenerator(BaseInputGenerator):
    """Generates inputs for the SwigluQuant operator."""

    names = ("SwigluQuant",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "x":
                out[name] = ctx.rand_tensor(name)
            elif name == "smooth_scales":
                out[name] = self._smooth_scales(ctx, out["x"])
            elif name == "offsets":
                out[name] = self._offsets(ctx, out.get("smooth_scales"))
            elif name == "group_index":
                out[name] = self._group_index(ctx, out["x"])
            elif name == "dst_type":
                out[name] = self._dst_type(ctx, input_spec)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _dst_type(self, ctx: InputGenerationContext, spec) -> Any:
        """Convert dst_type string to torch.dtype as expected by the NPU op."""
        if isinstance(spec, ScalarInput):
            value = spec.value
            if value == "int8":
                return torch.int8
            if value == "int4":
                return torch.quint4x2
        return None

    def _smooth_scales(
        self,
        ctx: InputGenerationContext,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Regenerate smooth_scales using the workload's mixed distribution."""
        return ctx.rand_tensor("smooth_scales")

    def _offsets(
        self,
        ctx: InputGenerationContext,
        smooth_scales: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Regenerate offsets; None when quant_mode is dynamic (1)."""
        quant_mode = self._scalar_value(ctx, "quant_mode")
        if quant_mode == 1:
            return None
        if smooth_scales is None:
            return None
        return ctx.rand_tensor("offsets")

    def _group_index(
        self,
        ctx: InputGenerationContext,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct non-decreasing group boundaries from x shape."""
        shape = ctx.resolve_shape("group_index")
        group_list_type = self._scalar_value(ctx, "group_list_type") or 0

        total_tokens = 1
        for s in x.shape[:-1]:
            total_tokens *= s
        num_groups = shape[0]
        group_size = total_tokens // num_groups if num_groups else total_tokens

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("group_index"))

        if group_list_type == 0:
            values = [(i + 1) * group_size for i in range(num_groups)]
        else:
            values = [group_size] * num_groups

        return torch.tensor(values, dtype=torch.int32, device=ctx.device)

    def _scalar_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None
