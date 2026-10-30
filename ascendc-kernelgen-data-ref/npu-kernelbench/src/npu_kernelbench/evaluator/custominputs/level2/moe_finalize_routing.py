"""Custom input generator for MoeFinalizeRouting.

Handles optional skip1/skip2/bias tensors (None when the workload shape is
None) and derives the expanded_src_to_dst_row index range from the shape of
expanded_permuted_rows.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class MoeFinalizeRoutingInputGenerator(BaseInputGenerator):
    """Generates inputs for the MoeFinalizeRouting operator."""

    names = ("MoeFinalizeRouting",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "expanded_permuted_rows":
                out[name] = ctx.rand_tensor(name)
            elif name in ("skip1", "skip2", "bias"):
                out[name] = self._optional_tensor(ctx, name)
            elif name == "scales":
                out[name] = self._scales_tensor(ctx)
            elif name == "expanded_src_to_dst_row":
                out[name] = self._make_expanded_src_to_dst_row(ctx, out["expanded_permuted_rows"])
            elif name == "export_for_source_row":
                out[name] = self._make_export_for_source_row(ctx)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _optional_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Return None if the workload omits the tensor, otherwise generate it."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        if getattr(input_spec, "shape", None) is None:
            return None
        return ctx.rand_tensor(name)

    def _scales_tensor(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Non-negative routing weights in [0, 1]."""
        shape = ctx.resolve_shape("scales")
        dtype = ctx.resolve_dtype("scales")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("scales"))
        return torch.rand(list(shape), generator=gen, dtype=torch.float32).to(device).to(dtype)

    def _make_expanded_src_to_dst_row(
        self,
        ctx: InputGenerationContext,
        expanded_permuted_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Generate index mapping in [0, expanded_permuted_rows.shape[0])."""
        shape = ctx.resolve_shape("expanded_src_to_dst_row")
        device = ctx.device
        upper = max(1, expanded_permuted_rows.shape[0])
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("expanded_src_to_dst_row"))
        return torch.randint(0, upper, list(shape), dtype=torch.int32, generator=gen).to(device)

    def _make_export_for_source_row(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Generate expert assignments in [0, 8)."""
        shape = ctx.resolve_shape("export_for_source_row")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("export_for_source_row"))
        return torch.randint(0, 8, list(shape), dtype=torch.int32, generator=gen).to(device)
