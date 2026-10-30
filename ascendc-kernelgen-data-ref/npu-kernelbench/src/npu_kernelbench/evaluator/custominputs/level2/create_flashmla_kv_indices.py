"""Custom input generator for CreateFlashmlaKvIndices."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class CreateFlashmlaKvIndicesInputGenerator(BaseInputGenerator):
    """Generates paged KV index tables for FlashMLA."""

    names = ("CreateFlashmlaKvIndices",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            spec = ctx.workload.inputs.get(name)
            if name == "req_to_token":
                out[name] = self._make_req_to_token(ctx)
            elif name == "req_pool_indices":
                out[name] = self._make_req_pool_indices(ctx)
            elif name == "page_kernel_lens":
                out[name] = self._make_page_kernel_lens(ctx)
            elif name == "kv_start_idx":
                out[name] = self._make_kv_start_idx(ctx)
            elif isinstance(spec, CustomInput) and getattr(spec, "value", None) is not None:
                out[name] = spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _seq_lens(self, ctx: InputGenerationContext) -> list[int]:
        spec = ctx.workload.inputs.get("page_kernel_lens")
        if isinstance(spec, CustomInput):
            if spec.value is not None:
                return list(spec.value)
            # Old JSON stored page_kernel_lens as a tensor shape only; generate
            # valid sequence lengths that fit inside max_context_len.
            shape = ctx.resolve_shape("page_kernel_lens")
            max_ctx = ctx.resolve_shape("req_to_token")[1]
            bs = shape[0]
            torch.manual_seed(ctx.derive_seed("page_kernel_lens"))
            return torch.randint(1, max(2, max_ctx), (bs,)).tolist()
        raise ValueError("page_kernel_lens spec is required")

    def _kv_start(self, ctx: InputGenerationContext) -> list[int] | None:
        spec = ctx.workload.inputs.get("kv_start_idx")
        if isinstance(spec, CustomInput):
            if spec.value is not None:
                val = spec.value
                if isinstance(val, list) and len(val) == 0:
                    return None
                return list(val)
            shape = ctx.resolve_shape("kv_start_idx")
            if shape[0] == 0:
                return None
            # Generate valid start offsets.
            max_ctx = ctx.resolve_shape("req_to_token")[1]
            seq_lens = self._seq_lens(ctx)
            torch.manual_seed(ctx.derive_seed("kv_start_idx"))
            return [int(torch.randint(0, max(1, max_ctx - sl + 1), (1,)).item()) for sl in seq_lens]
        return None

    def _page_size(self, ctx: InputGenerationContext) -> int:
        spec = ctx.workload.inputs.get("page_size")
        if isinstance(spec, CustomInput) and spec.value is not None:
            return int(spec.value)
        return 64

    def _make_req_to_token(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("req_to_token")
        dtype = ctx.resolve_dtype("req_to_token")
        device = ctx.device
        max_batch, max_ctx = shape
        page_size = self._page_size(ctx)

        seq_lens = self._seq_lens(ctx)
        kv_start = self._kv_start(ctx) or [0] * len(seq_lens)
        bs = len(seq_lens)

        req_to_token = torch.full((max_batch, max_ctx), -1, dtype=dtype, device=device)
        total_pages = max_ctx // page_size
        global_pages = torch.randperm(max_batch * total_pages, device=device).tolist()
        pg_off = 0
        for b in range(bs):
            start = kv_start[b]
            n_pages = (seq_lens[b] + page_size - 1) // page_size
            pages = global_pages[pg_off:pg_off + n_pages]
            pg_off += n_pages
            for j, pg in enumerate(pages):
                tok0 = start + j * page_size
                tok1 = min(tok0 + page_size, start + seq_lens[b])
                req_to_token[b, tok0:tok1] = torch.arange(
                    pg * page_size,
                    pg * page_size + (tok1 - tok0),
                    dtype=dtype,
                    device=device,
                )
        return req_to_token

    def _make_req_pool_indices(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("req_pool_indices")
        dtype = ctx.resolve_dtype("req_pool_indices")
        device = ctx.device
        max_batch = ctx.resolve_shape("req_to_token")[0]
        bs = len(self._seq_lens(ctx))
        return torch.randperm(max_batch, device=device)[:bs].to(dtype)

    def _make_page_kernel_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("page_kernel_lens")
        dtype = ctx.resolve_dtype("page_kernel_lens")
        device = ctx.device
        return torch.tensor(self._seq_lens(ctx), dtype=dtype, device=device)

    def _make_kv_start_idx(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("kv_start_idx")
        dtype = ctx.resolve_dtype("kv_start_idx")
        device = ctx.device
        kv_start = self._kv_start(ctx)
        if kv_start is None:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.tensor(kv_start, dtype=dtype, device=device)
