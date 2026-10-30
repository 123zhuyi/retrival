"""Custom input generator for TrtllmBatchDecodeWithKvCacheMla."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class TrtllmBatchDecodeWithKvCacheMlaInputGenerator(BaseInputGenerator):
    """Generates TRTLLM batch decode MLA paged indices."""

    names = ("TrtllmBatchDecodeWithKvCacheMla",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name in ("query", "kv_cache"):
                out[name] = ctx.rand_tensor(name)
            elif name == "block_tables":
                out[name] = self._make_block_tables(ctx)
            elif name == "seq_lens":
                out[name] = self._make_seq_lens(ctx)
            elif name == "cum_seq_lens_q":
                out[name] = self._make_cum_seq_lens_q(ctx)
            elif name in ("kv_lora_rank", "qk_rope_head_dim", "bmm1_scale", "bmm2_scale", "sparse_mla_top_k", "return_lse", "return_lse_base_on_e"):
                out[name] = self._get_scalar(ctx, name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _kv_lens(self, ctx: InputGenerationContext) -> list[int]:
        return list(self._get_scalar(ctx, "kv_lens", []))

    def _qo_lens(self, ctx: InputGenerationContext) -> list[int] | None:
        value = self._get_scalar(ctx, "qo_lens", None)
        if value is None:
            return None
        return list(value)

    def _page_size(self, ctx: InputGenerationContext) -> int:
        return int(self._get_scalar(ctx, "page_size", 0))

    def _make_block_tables(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        page_size = self._page_size(ctx)
        device = ctx.device
        shape = ctx.resolve_shape("block_tables")
        total_pages = ctx.resolve_shape("kv_cache")[0]
        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        sparse_mla_top_k = int(self._get_scalar(ctx, "sparse_mla_top_k", 0))

        if sparse_mla_top_k > 0:
            B, q_len, top_k = shape
            block_tables = torch.zeros(B, q_len, top_k, dtype=torch.int32, device=device)
            for b in range(B):
                for t in range(q_len):
                    n_valid = top_k - (b + t) % 2
                    pages = torch.randperm(total_pages, device=device)[:n_valid]
                    block_tables[b, t, :n_valid] = pages.to(torch.int32)
                    block_tables[b, t, n_valid:] = -1
            return block_tables
        else:
            B, max_pages = shape
            perm = torch.randperm(total_pages, dtype=torch.int32, device=device)
            block_tables = torch.zeros(B, max_pages, dtype=torch.int32, device=device)
            off = 0
            for b, n in enumerate(npages):
                block_tables[b, :n] = perm[off:off + n]
                off += n
            return block_tables

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        device = ctx.device
        return torch.tensor(kv_lens, dtype=torch.int32, device=device)

    def _make_cum_seq_lens_q(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        qo_lens = self._qo_lens(ctx)
        if qo_lens is None:
            return None
        device = ctx.device
        indptr = [0]
        for l in qo_lens:
            indptr.append(indptr[-1] + l)
        return torch.tensor(indptr, dtype=torch.int32, device=device)
