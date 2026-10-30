"""Custom input generator for MultimodalRopePositionComputationWithGridBasedIndexing.

Mirrors the old benchmark: ``grid_thw`` dimensions are multiples of
``spatial_merge_size`` so the reference division produces integer patch
grids, and the float tensors use the mixed normal/uniform distribution.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class MultimodalRopePositionComputationWithGridBasedIndexingInputGenerator(BaseInputGenerator):
    """Generates inputs for multimodal RoPE position computation."""

    names = ("MultimodalRopePositionComputationWithGridBasedIndexing",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "grid_thw":
                out[name] = self._make_grid_thw(ctx)
            elif name in ("pos_embed_weight", "inv_freq"):
                out[name] = self._mixed_float_tensor(ctx, name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_grid_thw(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Grid dimensions are multiples of spatial_merge_size."""
        shape = ctx.resolve_shape("grid_thw")
        dtype = ctx.resolve_dtype("grid_thw")
        merge_size = int(self._scalar_value(ctx, "spatial_merge_size") or 2)
        device = ctx.device

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("grid_thw"))
        values = torch.randint(1, 8, list(shape), dtype=dtype, generator=gen).to(device)
        return values * merge_size

    def _mixed_float_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor:
        """Legacy mixed distribution: whole case is normal or uniform."""
        return ctx.rand_tensor(name, distribution="mixed")

    def _scalar_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None
