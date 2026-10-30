"""Custom input generator for MultiMaskAttentionAggregation.

Forces ref_target_masks to strict 0/1 float32 values via boolean cast.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class MultiMaskAttentionAggregationInputGenerator(BaseInputGenerator):
    """Generates inputs for the MultiMaskAttentionAggregation operator."""

    names = ("MultiMaskAttentionAggregation",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "ref_target_masks":
                out[name] = self._make_masks(ctx)
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_masks(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Generate strict 0/1 float32 masks from boolean thresholds."""
        shape = ctx.resolve_shape("ref_target_masks")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("ref_target_masks"))
        return (torch.rand(list(shape), generator=gen).to(device) > 0.5).to(torch.float32)
