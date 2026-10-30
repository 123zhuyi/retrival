"""Custom input generator for MultiHeadLatentAttention."""

from __future__ import annotations

from typing import Any

import torch

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class MultiHeadLatentAttentionInputGenerator(BaseInputGenerator):
    """Generates inputs for the MultiHeadLatentAttention operator."""

    names = ("MultiHeadLatentAttention",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        q = ctx.rand_tensor("q")
        kv_cache = ctx.rand_tensor("kv_cache")

        batch, query_length, _, qk_dim = q.shape
        num_blocks, page_size, _, _ = kv_cache.shape

        page_table_shape = ctx.resolve_shape("block_table")
        _, num_pages = page_table_shape
        page_table_dtype = ctx.resolve_dtype("block_table")

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("block_table"))
        block_table = torch.stack([
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

        headdim_v = ctx.scalar("headdim_v")
        causal = ctx.scalar("causal")

        out = {
            "q": q,
            "kv_cache": kv_cache,
            "block_table": block_table,
            "cache_seqlens": cache_seqlens,
            "headdim_v": headdim_v,
            "causal": causal,
        }
        return [out[name] for name in ctx.definition.inputs]
