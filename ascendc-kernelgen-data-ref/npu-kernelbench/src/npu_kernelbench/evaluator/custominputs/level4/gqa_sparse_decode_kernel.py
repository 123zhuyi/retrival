"""Custom input generator for GqaSparseDecodeKernel."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


SPARSE_BLOCK_SIZE = 128


class GqaSparseDecodeKernelInputGenerator(BaseInputGenerator):
    """Generates paged KV cache and block tables for GQA sparse decode (level4 kernel)."""

    names = ("GqaSparseDecodeKernel",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "q":
                out[name] = ctx.rand_tensor(name)
            elif name == "kv_cache":
                out[name] = ctx.rand_tensor(name)
            elif name == "topk_idx":
                out[name] = self._make_topk_idx(ctx)
            elif name == "block_table":
                out[name] = self._make_block_table(ctx)
            elif name == "seq_lens":
                out[name] = self._make_seq_lens(ctx)
            elif name == "sm_scale":
                out[name] = self._get_scalar(ctx, "sm_scale")
            elif name == "decode_query_len":
                out[name] = self._get_scalar(ctx, "decode_query_len")
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _decode_query_len(self, ctx: InputGenerationContext) -> int:
        value = self._get_scalar(ctx, "decode_query_len")
        if value is None:
            raise ValueError("decode_query_len scalar is required")
        return int(value)

    def _batch_size(self, ctx: InputGenerationContext) -> int:
        total_q = ctx.resolve_shape("q")[0]
        dql = self._decode_query_len(ctx)
        if dql <= 0 or total_q % dql != 0:
            raise ValueError(f"total_q {total_q} must be divisible by decode_query_len {dql}")
        return total_q // dql

    def _compute_nblocks(self, ctx: InputGenerationContext) -> list[int]:
        P = ctx.resolve_shape("kv_cache")[0]
        B, max_block = ctx.resolve_shape("block_table")
        dql = self._decode_query_len(ctx)
        min_per_req = max(1, (dql + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE)
        available = max(0, P - 3)
        target_total = min(available, B * max_block)
        target_total = max(target_total, B * min_per_req)
        base = target_total // B
        rem = target_total % B
        nblocks = [base + (1 if i < rem else 0) for i in range(B)]
        return nblocks

    def _make_block_table(self, ctx: InputGenerationContext) -> torch.Tensor:
        nblocks = self._compute_nblocks(ctx)
        P = ctx.resolve_shape("kv_cache")[0]
        dtype = torch.int32
        device = ctx.device
        B = len(nblocks)
        max_block = ctx.resolve_shape("block_table")[1]

        perm = torch.randperm(P, dtype=dtype, device=device)
        block_table = torch.zeros(B, max_block, dtype=dtype, device=device)
        off = 0
        for b, n in enumerate(nblocks):
            block_table[b, :n] = perm[off:off + n]
            off += n
        return block_table

    def _make_topk_idx(self, ctx: InputGenerationContext) -> torch.Tensor:
        nblocks = self._compute_nblocks(ctx)
        dql = self._decode_query_len(ctx)
        B = self._batch_size(ctx)
        seq_lens = [n * SPARSE_BLOCK_SIZE for n in nblocks]
        shape = ctx.resolve_shape("topk_idx")
        dtype = torch.int32
        device = ctx.device
        KH, total_q, topk = shape

        topk_idx = torch.zeros(KH, total_q, topk, dtype=dtype, device=device)
        for pid_b in range(total_q):
            req_id = pid_b // dql
            q_offset = pid_b - req_id * dql
            kv_len = seq_lens[req_id] - dql + q_offset + 1
            num_blocks = (kv_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
            num_blocks = min(num_blocks, nblocks[req_id])
            n_sel = min(topk, num_blocks)
            for kh in range(KH):
                if n_sel > 0:
                    sel = torch.randperm(num_blocks, device=device)[:n_sel].sort().values.to(dtype)
                    topk_idx[kh, pid_b, :n_sel] = sel
                if n_sel < topk:
                    topk_idx[kh, pid_b, n_sel:] = torch.randint(
                        0, max(num_blocks, 1), (topk - n_sel,), dtype=dtype, device=device
                    )
        return topk_idx

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        nblocks = self._compute_nblocks(ctx)
        seq_lens = [n * SPARSE_BLOCK_SIZE for n in nblocks]
        return torch.tensor(seq_lens, dtype=torch.int32, device=ctx.device)
