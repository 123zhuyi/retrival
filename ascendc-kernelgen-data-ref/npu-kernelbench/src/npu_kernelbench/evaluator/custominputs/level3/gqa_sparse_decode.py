"""Custom input generator for GqaSparseDecode."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


SPARSE_BLOCK_SIZE = 128


class GqaSparseDecodeInputGenerator(BaseInputGenerator):
    """Generates paged KV cache and block tables for GQA sparse decode."""

    names = ("GqaSparseDecode",)
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

    def _seq_lens(self, ctx: InputGenerationContext) -> list[int]:
        value = self._get_value(ctx, "seq_lens")
        if value is None:
            raise ValueError("seq_lens value is required")
        return list(value)

    def _decode_query_len(self, ctx: InputGenerationContext) -> int:
        value = self._get_value(ctx, "decode_query_len")
        if value is None:
            raise ValueError("decode_query_len value is required")
        return int(value)

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

        seq_lens = self._seq_lens(ctx)
        dql = self._decode_query_len(ctx)
        B = len(seq_lens)
        if total_q != B * dql:
            raise ValueError(f"topk_idx total_q {total_q} != B*dql={B*dql}")

        topk_idx = torch.zeros(KH, total_q, topk, dtype=dtype, device=device)
        for pid_b in range(total_q):
            req_id = pid_b // dql
            q_offset = pid_b - req_id * dql
            kv_len = seq_lens[req_id] - dql + q_offset + 1
            num_blocks = (kv_len + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE
            n_sel = min(topk, num_blocks)
            for kh in range(KH):
                sel = torch.randperm(num_blocks, device=device)[:n_sel].sort().values.to(dtype)
                topk_idx[kh, pid_b, :n_sel] = sel
                if n_sel < topk:
                    topk_idx[kh, pid_b, n_sel:] = torch.randint(
                        0, num_blocks, (topk - n_sel,), dtype=dtype, device=device
                    )
        return topk_idx

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        dtype = torch.int32
        device = ctx.device
        return torch.tensor(self._seq_lens(ctx), dtype=dtype, device=device)
