"""Custom input generator for BatchDecodeMlaPaged."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class BatchDecodeMlaPagedInputGenerator(BaseInputGenerator):
    """Generates paged KV indices for batch decode MLA attention."""

    names = ("BatchDecodeMlaPaged",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name in ("q_nope", "q_pe", "paged_ckv_cache", "paged_kpe_cache"):
                out[name] = ctx.rand_tensor(name)
            elif name == "kv_indptr":
                out[name] = self._make_kv_indptr(ctx)
            elif name == "kv_indices":
                out[name] = self._make_kv_indices(ctx)
            elif name == "kv_last_page_len":
                out[name] = self._make_kv_last_page_len(ctx)
            elif name in ("sm_scale", "window_left", "logits_soft_cap", "return_lse"):
                out[name] = self._get_scalar(ctx, name, default=self._default_for(name))
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _default_for(self, name: str) -> Any:
        defaults = {
            "window_left": -1,
            "logits_soft_cap": 0.0,
            "return_lse": False,
        }
        return defaults.get(name, None)

    def _kv_lens(self, ctx: InputGenerationContext) -> list[int]:
        return list(self._get_scalar(ctx, "kv_lens", []))

    def _page_size(self, ctx: InputGenerationContext) -> int:
        return int(self._get_scalar(ctx, "page_size", 0))

    def _make_kv_indptr(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        page_size = self._page_size(ctx)
        device = ctx.device
        indptr = [0]
        for l in kv_lens:
            n = (l + page_size - 1) // page_size
            indptr.append(indptr[-1] + n)
        return torch.tensor(indptr, dtype=torch.int32, device=device)

    def _make_kv_indices(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        page_size = self._page_size(ctx)
        device = ctx.device
        total_pages = ctx.resolve_shape("paged_ckv_cache")[0]
        npages = [(l + page_size - 1) // page_size for l in kv_lens]
        perm = torch.randperm(total_pages, dtype=torch.int32, device=device)
        indices = []
        off = 0
        for n in npages:
            indices.extend(perm[off:off + n].tolist())
            off += n
        return torch.tensor(indices, dtype=torch.int32, device=device)

    def _make_kv_last_page_len(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        page_size = self._page_size(ctx)
        device = ctx.device
        last_len = []
        for l in kv_lens:
            n = (l + page_size - 1) // page_size
            last_len.append(l - (n - 1) * page_size)
        return torch.tensor(last_len, dtype=torch.int32, device=device)
