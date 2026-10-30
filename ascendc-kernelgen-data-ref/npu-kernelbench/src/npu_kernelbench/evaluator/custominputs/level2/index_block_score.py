"""Custom input generator for IndexBlockScore."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


SPARSE_BLOCK_SIZE = 128


class IndexBlockScoreInputGenerator(BaseInputGenerator):
    """Generates paged index-K cache and block tables for prefill index scoring."""

    names = ("IndexBlockScore",)
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
            elif name == "cu_seqlens":
                out[name] = self._make_cu_seqlens(ctx)
            elif name == "seq_lens":
                out[name] = self._make_seq_lens(ctx)
            elif name == "prefix_lens":
                out[name] = self._make_prefix_lens(ctx)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _q_lens(self, ctx: InputGenerationContext) -> list[int]:
        spec = ctx.workload.inputs.get("q_lens")
        if isinstance(spec, CustomInput):
            return list(spec.value)
        raise ValueError("q_lens value is required")

    def _prefix_lens(self, ctx: InputGenerationContext) -> list[int]:
        spec = ctx.workload.inputs.get("prefix_lens")
        if isinstance(spec, CustomInput):
            return list(spec.value)
        raise ValueError("prefix_lens value is required")

    def _make_block_table(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("block_table")
        dtype = torch.int32
        device = ctx.device
        B, max_block = shape
        P = ctx.resolve_shape("index_kv_cache")[0]

        q_lens = self._q_lens(ctx)
        prefix_lens = self._prefix_lens(ctx)
        seq_lens = [p + ql for p, ql in zip(prefix_lens, q_lens)]
        needed = [(sl + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE for sl in seq_lens]
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

    def _make_cu_seqlens(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        q_lens = self._q_lens(ctx)
        cu = torch.zeros(len(q_lens) + 1, dtype=dtype, device=device)
        cu[1:] = torch.cumsum(torch.tensor(q_lens, dtype=dtype, device=device), dim=0)
        return cu

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        prefix_lens = self._prefix_lens(ctx)
        q_lens = self._q_lens(ctx)
        seq_lens = [p + ql for p, ql in zip(prefix_lens, q_lens)]
        return torch.tensor(seq_lens, dtype=dtype, device=device)

    def _make_prefix_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        return torch.tensor(self._prefix_lens(ctx), dtype=dtype, device=device)
