"""Custom input generator for DecodeIndexScore."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


SPARSE_BLOCK_SIZE = 128


class DecodeIndexScoreInputGenerator(BaseInputGenerator):
    """Generates paged index-K cache and block tables for decode index scoring."""

    names = ("DecodeIndexScore",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "idx_q":
                out[name] = ctx.rand_tensor(name)
            elif name == "index_kv_cache":
                out[name] = ctx.rand_tensor(name)
            elif name == "block_table":
                out[name] = self._make_block_table(ctx)
            elif name == "seq_lens":
                out[name] = self._make_seq_lens(ctx)
            elif isinstance(spec := ctx.workload.inputs.get(name), CustomInput) and spec.value is not None:
                out[name] = spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _seq_lens(self, ctx: InputGenerationContext) -> list[int]:
        spec = ctx.workload.inputs.get("seq_lens")
        if isinstance(spec, CustomInput):
            return list(spec.value)
        raise ValueError("seq_lens value is required")

    def _decode_query_len(self, ctx: InputGenerationContext) -> int:
        spec = ctx.workload.inputs.get("decode_query_len")
        if isinstance(spec, CustomInput) and spec.value is not None:
            return int(spec.value)
        raise ValueError("decode_query_len value is required")

    def _make_block_table(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("block_table")
        dtype = torch.int32
        device = ctx.device
        B, max_block = shape
        P = ctx.resolve_shape("index_kv_cache")[0]

        seq_lens = self._seq_lens(ctx)
        dql = self._decode_query_len(ctx)
        total_q = B * dql
        q_shape = ctx.resolve_shape("idx_q")
        if q_shape[0] != total_q:
            raise ValueError(f"idx_q shape {q_shape} inconsistent with B={B} * dql={dql}")

        needed = [(s + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE for s in seq_lens]
        if max(needed) > max_block:
            raise ValueError(f"max_block {max_block} is too small, need {max(needed)}")
        if P < sum(needed) + 3:
            raise ValueError(f"num_pages {P} is too small, need {sum(needed) + 3}")

        perm = torch.randperm(P, dtype=dtype, device=device)
        block_table = torch.zeros(B, max_block, dtype=dtype, device=device)
        off = 0
        sum_needed = sum(needed)
        unused = P - sum_needed
        for b in range(B):
            nb = needed[b]
            block_table[b, :nb] = perm[off:off + nb]
            off += nb
            for j in range(nb, max_block):
                block_table[b, j] = perm[sum_needed + (j - nb) % unused]
        return block_table

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        return torch.tensor(self._seq_lens(ctx), dtype=dtype, device=device)
