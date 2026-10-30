"""Custom input generator for KvRmsnormRopeCache.

Reproduces mixed float generation and cache-mode-dependent valid index values.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class KvRmsnormRopeCacheInputGenerator(BaseInputGenerator):
    """Generates inputs for the KvRmsnormRopeCache operator."""

    names = ("KvRmsnormRopeCache",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        # First pass: generate all random/scalar inputs so that cache tensors
        # are available when we compute valid indices.
        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            else:
                out[name] = None

        # Second pass: regenerate index based on the generated k_cache.
        if "index" in ctx.definition.inputs:
            out["index"] = self._make_index(ctx, out.get("k_cache"))

        return [out[name] for name in ctx.definition.inputs]

    def _make_index(
        self,
        ctx: InputGenerationContext,
        k_cache: torch.Tensor | None,
    ) -> torch.Tensor:
        """Generate valid cache write indices based on cache_mode."""
        shape = ctx.resolve_shape("index")
        cache_mode = self._scalar_value(ctx, "cache_mode") or "Norm"
        device = ctx.device

        if k_cache is None:
            return torch.zeros(list(shape), dtype=torch.int64, device=device)

        total = 1
        for s in shape:
            total *= s

        if cache_mode == "Norm":
            max_seq = k_cache.shape[2]
            values = (torch.arange(total, device=device) % max_seq).to(torch.int64)
        elif cache_mode in ("PA", "PA_BNSD", "PA_NZ"):
            values = torch.arange(total, dtype=torch.int64, device=device)
        elif cache_mode in ("PA_BLK_BNSD", "PA_BLK_NZ"):
            block_size = k_cache.shape[1]
            values = torch.arange(total, dtype=torch.int64, device=device) * block_size
        else:
            values = torch.arange(total, dtype=torch.int64, device=device)

        return values.reshape(shape)

    def _scalar_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None
