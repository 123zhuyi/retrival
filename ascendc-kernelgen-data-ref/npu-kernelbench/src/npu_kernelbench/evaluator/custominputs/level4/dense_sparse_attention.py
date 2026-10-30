"""Custom input generator for DenseSparseAttention."""

from __future__ import annotations

from typing import Any

import torch

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class DenseSparseAttentionInputGenerator(BaseInputGenerator):
    """Generates inputs for the DenseSparseAttention operator."""

    names = ("DenseSparseAttention",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        q = ctx.rand_tensor("q")
        kv_cache = ctx.rand_tensor("kv_cache")

        num_blocks, page_size, _, _ = kv_cache.shape
        indices_shape = ctx.resolve_shape("indices")
        indices_dtype = ctx.resolve_dtype("indices")

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("indices"))
        max_index = num_blocks * page_size
        indices = torch.randint(
            0, max_index, list(indices_shape), dtype=indices_dtype, device=ctx.device, generator=gen
        )

        headdim_v = ctx.scalar("headdim_v")

        out = {
            "q": q,
            "kv_cache": kv_cache,
            "indices": indices,
            "headdim_v": headdim_v,
        }
        return [out[name] for name in ctx.definition.inputs]
