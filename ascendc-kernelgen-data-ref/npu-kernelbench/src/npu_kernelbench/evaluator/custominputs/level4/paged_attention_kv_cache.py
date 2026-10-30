"""Custom input generator for PagedAttentionKVCache.

Generates q/k_cache/v_cache randomly and creates consistent page_table
and cache_seqlens matching the old benchmark.
"""

from __future__ import annotations

from typing import Any

import torch

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class PagedAttentionKVCacheInputGenerator(BaseInputGenerator):
    """Generates inputs for the PagedAttentionKVCache operator."""

    names = ("PagedAttentionKVCache",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        q = ctx.rand_tensor("q")
        k_cache = ctx.rand_tensor("k_cache")
        v_cache = ctx.rand_tensor("v_cache")

        batch, query_length, n_heads, head_dim = q.shape
        num_blocks, page_size, n_kv_heads, _ = k_cache.shape

        page_table_shape = ctx.resolve_shape("page_table")
        _, num_pages = page_table_shape
        page_table_dtype = ctx.resolve_dtype("page_table")

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("page_table"))
        page_table = torch.stack([
            torch.randperm(num_blocks, generator=gen)[:num_pages]
            for _ in range(batch)
        ]).to(dtype=page_table_dtype, device=ctx.device)

        cache_seqlens_shape = ctx.resolve_shape("cache_seqlens")
        cache_seqlens_dtype = ctx.resolve_dtype("cache_seqlens")
        seed = ctx.derive_seed("cache_seqlens")
        capacity = num_pages * page_size
        values = []
        for row in range(batch):
            reduction = (seed * 7 + row * 3) % max(1, page_size)
            values.append(max(int(query_length), capacity - reduction))
        cache_seqlens = torch.tensor(values, dtype=cache_seqlens_dtype, device=ctx.device)

        causal = ctx.scalar("causal")

        out = {
            "q": q,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "cache_seqlens": cache_seqlens,
            "page_table": page_table,
            "causal": causal,
        }
        return [out[name] for name in ctx.definition.inputs]
