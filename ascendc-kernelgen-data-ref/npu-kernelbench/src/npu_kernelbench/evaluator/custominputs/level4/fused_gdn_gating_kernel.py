"""Custom input generator for FusedGdnGatingKernel."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusedGdnGatingKernelInputGenerator(BaseInputGenerator):
    """Generates GDN gating inputs in physically stable ranges."""

    names = ("FusedGdnGatingKernel",)
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
            elif name in ("beta", "threshold"):
                out[name] = self._get_scalar(ctx, name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

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
