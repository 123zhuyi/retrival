"""Custom input generator for FusedKdaGate."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusedKdaGateInputGenerator(BaseInputGenerator):
    """Generates fused KDA gate inputs."""

    names = ("FusedKdaGate",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "g":
                out[name] = self._make_scaled(ctx, name, 0.9)
            elif name == "A":
                out[name] = self._make_scaled(ctx, name, 0.4)
            elif name == "g_bias":
                out[name] = self._make_scaled(ctx, name, 0.3)
            elif name in ("head_k_dim", "beta", "threshold", "lower_bound", "use_bias", "use_lower_bound"):
                out[name] = self._get_scalar(ctx, name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _make_scaled(self, ctx: InputGenerationContext, name: str, scale: float) -> torch.Tensor:
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        torch.manual_seed(ctx.derive_seed(name))
        t = torch.randn(shape, dtype=torch.float32, device=device) * scale
        return t.to(dtype)
