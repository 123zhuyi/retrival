"""Custom input generator for GqaSparseFwd."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


SPARSE_BLOCK_SIZE = 128


class GqaSparseFwdInputGenerator(BaseInputGenerator):
    """Generates paged KV cache and block tables for GQA sparse prefill."""

    names = ("GqaSparseFwd",)
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
            elif isinstance(spec := ctx.workload.inputs.get(name), CustomInput) and spec.value is not None:
                out[name] = spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, CustomInput):
            return spec.value
        return None

    def _cu_seqlens_q(self, ctx: InputGenerationContext) -> list[int]:
        value = self._get_value(ctx, "cu_seqlens_q")
        if value is None:
            raise ValueError("cu_seqlens_q value is required")
        return list(value)

    def _prefix_lens(self, ctx: InputGenerationContext) -> list[int]:
        value = self._get_value(ctx, "prefix_lens")
        if value is None:
            raise ValueError("prefix_lens value is required")
        return list(value)

    def _q_lens(self, ctx: InputGenerationContext) -> list[int]:
        cu = self._cu_seqlens_q(ctx)
        return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]

    def _make_block_table(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("block_table")
        dtype = torch.int32
        device = ctx.device
        B, max_block = shape
        P = ctx.resolve_shape("kv_cache")[0]

        seq_lens = self._seq_lens(ctx)
        nblocks = [(s + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE for s in seq_lens]
        total_pages = sum(nblocks) + 3
        if P != total_pages:
            raise ValueError(f"kv_cache num_blocks {P} != expected {total_pages}")
        if max(nblocks) > max_block:
            raise ValueError(f"max_block {max_block} too small, need {max(nblocks)}")

        perm = torch.randperm(P, dtype=dtype, device=device)
        block_table = torch.zeros(B, max_block, dtype=dtype, device=device)
        off = 0
        for b, n in enumerate(nblocks):
            block_table[b, :n] = perm[off:off + n]
            off += n
        return block_table

    def _make_topk_idx(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("topk_idx")
        dtype = torch.int32
        device = ctx.device
        KH, total_q, topk = shape

        cu = self._cu_seqlens_q(ctx)
        prefix_lens = self._prefix_lens(ctx)
        q_lens = self._q_lens(ctx)
        B = len(q_lens)

        topk_idx = torch.zeros(KH, total_q, topk, dtype=dtype, device=device)
        for b in range(B):
            qs = cu[b]
            for j in range(q_lens[b]):
                q_abs = prefix_lens[b] + j
                valid_blocks = (q_abs + SPARSE_BLOCK_SIZE) // SPARSE_BLOCK_SIZE
                n_sel = min(topk, valid_blocks)
                for kh in range(KH):
                    sel = torch.randperm(valid_blocks, device=device)[:n_sel].sort().values.to(dtype)
                    topk_idx[kh, qs + j, :n_sel] = sel
                    if n_sel < topk:
                        topk_idx[kh, qs + j, n_sel:] = torch.randint(
                            0, valid_blocks, (topk - n_sel,), dtype=dtype, device=device
                        )
        return topk_idx

    def _make_cu_seqlens_q(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        return torch.tensor(self._cu_seqlens_q(ctx), dtype=dtype, device=device)

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        return torch.tensor(self._seq_lens(ctx), dtype=dtype, device=device)

    def _make_prefix_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        return torch.tensor(self._prefix_lens(ctx), dtype=dtype, device=device)

    def _seq_lens(self, ctx: InputGenerationContext) -> list[int]:
        return [p + ql for p, ql in zip(self._prefix_lens(ctx), self._q_lens(ctx))]
