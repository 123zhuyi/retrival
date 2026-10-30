"""Custom input generator for FusedGdnGating."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusedGdnGatingInputGenerator(BaseInputGenerator):
    """Generates GDN gating inputs in physically stable ranges."""

    names = ("FusedGdnGating",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "A_log":
                out[name] = self._make_clamped(ctx, "A_log", -10.0, 5.0)
            elif name in ("a", "b"):
                out[name] = ctx.rand_tensor(name)
            elif name == "dt_bias":
                out[name] = self._make_clamped(ctx, "dt_bias", -5.0, 5.0)
            elif isinstance(spec := ctx.workload.inputs.get(name), CustomInput) and spec.value is not None:
                out[name] = spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_clamped(
        self,
        ctx: InputGenerationContext,
        name: str,
        low: float,
        high: float,
    ) -> torch.Tensor:
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        return ctx.rand_tensor(name, shape=shape, dtype=torch.float32).clamp(low, high).to(dtype)
