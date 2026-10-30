"""Custom input generator for FusedRopeWithQkNormAndKvCacheUpdate.

Overrides cache_position so it always indexes valid positions in the cache.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusedRopeWithQkNormAndKvCacheUpdateInputGenerator(BaseInputGenerator):
    """Generates inputs for the FusedRopeWithQkNormAndKvCacheUpdate operator."""

    names = ("FusedRopeWithQkNormAndKvCacheUpdate",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "cache_position":
                out[name] = self._make_cache_position(ctx, out["value_cache"])
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_cache_position(
        self,
        ctx: InputGenerationContext,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Generate cache positions in [0, value_cache.shape[2])."""
        shape = ctx.resolve_shape("cache_position")
        dtype = ctx.resolve_dtype("cache_position")
        device = ctx.device
        max_pos = value_cache.shape[2]
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("cache_position"))
        return torch.randint(0, max_pos, list(shape), dtype=dtype, generator=gen).to(device)
