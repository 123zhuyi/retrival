"""Custom input generator for GqaSparseFwdKernel."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


SPARSE_BLOCK_SIZE = 128


class GqaSparseFwdKernelInputGenerator(BaseInputGenerator):
    """Generates paged KV cache and block tables for GQA sparse prefill (level4 kernel)."""

    names = ("GqaSparseFwdKernel",)
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
            elif name == "cu_seqlens_q":
                out[name] = self._make_cu_seqlens_q(ctx)
            elif name == "seq_lens":
                out[name] = self._make_seq_lens(ctx)
            elif name == "prefix_lens":
                out[name] = self._make_prefix_lens(ctx)
            elif name == "sm_scale":
                out[name] = self._get_scalar(ctx, "sm_scale")
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _compute_q_lens(self, ctx: InputGenerationContext) -> tuple[list[int], list[int]]:
        total_q = ctx.resolve_shape("q")[0]
        cu_shape = ctx.resolve_shape("cu_seqlens_q")
        B = cu_shape[0] - 1
        if B <= 0:
            raise ValueError(f"Invalid cu_seqlens_q shape {cu_shape}")
        base = total_q // B
        rem = total_q % B
        q_lens = [base + (1 if i < rem else 0) for i in range(B)]
        cu = [0]
        for l in q_lens:
            cu.append(cu[-1] + l)
        return q_lens, cu

    def _compute_nblocks(self, ctx: InputGenerationContext) -> tuple[list[int], int, int]:
        P = ctx.resolve_shape("kv_cache")[0]
        B, max_block = ctx.resolve_shape("block_table")
        available = max(0, P - 3)
        target_total = min(available, B * max_block)
        target_total = max(target_total, B)  # at least one block per request
        base = target_total // B
        rem = target_total % B
        nblocks = [base + (1 if i < rem else 0) for i in range(B)]
        return nblocks, P, max_block

    def _make_block_table(self, ctx: InputGenerationContext) -> torch.Tensor:
        nblocks, P, _ = self._compute_nblocks(ctx)
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
        q_lens, cu = self._compute_q_lens(ctx)
        nblocks, _, _ = self._compute_nblocks(ctx)
        shape = ctx.resolve_shape("topk_idx")
        dtype = torch.int32
        device = ctx.device
        KH, total_q, topk = shape
        B = len(q_lens)

        topk_idx = torch.zeros(KH, total_q, topk, dtype=dtype, device=device)
        for b in range(B):
            qs = cu[b]
            for j in range(q_lens[b]):
                q_abs = j  # prefix_lens is kept at 0
                valid_blocks = (q_abs + SPARSE_BLOCK_SIZE) // SPARSE_BLOCK_SIZE
                valid_blocks = min(valid_blocks, nblocks[b])
                n_sel = min(topk, valid_blocks)
                for kh in range(KH):
                    if n_sel > 0:
                        sel = torch.randperm(valid_blocks, device=device)[:n_sel].sort().values.to(dtype)
                        topk_idx[kh, qs + j, :n_sel] = sel
                    if n_sel < topk:
                        topk_idx[kh, qs + j, n_sel:] = torch.randint(
                            0, max(valid_blocks, 1), (topk - n_sel,), dtype=dtype, device=device
                        )
        return topk_idx

    def _make_cu_seqlens_q(self, ctx: InputGenerationContext) -> torch.Tensor:
        _, cu = self._compute_q_lens(ctx)
        return torch.tensor(cu, dtype=torch.int32, device=ctx.device)

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        q_lens, _ = self._compute_q_lens(ctx)
        nblocks, _, _ = self._compute_nblocks(ctx)
        seq_lens = [n * SPARSE_BLOCK_SIZE for n in nblocks]
        return torch.tensor(seq_lens, dtype=torch.int32, device=ctx.device)

    def _make_prefix_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        B = ctx.resolve_shape("cu_seqlens_q")[0] - 1
        return torch.zeros(B, dtype=torch.int32, device=ctx.device)
