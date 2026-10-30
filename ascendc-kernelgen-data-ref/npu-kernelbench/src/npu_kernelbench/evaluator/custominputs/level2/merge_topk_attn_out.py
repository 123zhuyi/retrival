"""Custom input generator for MergeTopkAttnOut."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class MergeTopkAttnOutInputGenerator(BaseInputGenerator):
    """Generates split-K partial attention outputs and log2-domain LSE values."""

    names = ("MergeTopkAttnOut",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            spec = ctx.workload.inputs.get(name)
            if name == "o_partial":
                out[name] = ctx.rand_tensor(name)
            elif name == "lse_partial":
                out[name] = self._make_lse(ctx)
            elif isinstance(spec, CustomInput) and getattr(spec, "value", None) is not None:
                out[name] = spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_lse(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("lse_partial")
        dtype = ctx.resolve_dtype("lse_partial")
        device = ctx.device
        C, total_q, H = shape

        lse = torch.empty(shape, dtype=torch.float32, device=device)
        lse.uniform_(-10.0, 20.0)
        # Each row has at least one valid chunk; remaining chunks are -inf.
        n_valid = torch.randint(1, C + 1, (total_q, H), device=device)
        chunk_ids = torch.arange(C, device=device).view(C, 1, 1)
        lse = torch.where(
            chunk_ids < n_valid.view(1, total_q, H),
            lse,
            torch.full_like(lse, float("-inf")),
        )
        return lse.to(dtype)
