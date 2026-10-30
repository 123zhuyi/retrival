"""Custom input generator for TanhGatedResidualAddBackward.

Forces the mask to strict 0/1 bfloat16 values via integer cast.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class TanhGatedResidualAddBackwardInputGenerator(BaseInputGenerator):
    """Generates inputs for the TanhGatedResidualAddBackward operator."""

    names = ("TanhGatedResidualAddBackward",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "mask":
                out[name] = self._make_mask(ctx)
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_mask(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Generate strict 0/1 mask with the target dtype."""
        shape = ctx.resolve_shape("mask")
        dtype = ctx.resolve_dtype("mask")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("mask"))
        return torch.randint(0, 2, list(shape), dtype=dtype, generator=gen).to(device)
