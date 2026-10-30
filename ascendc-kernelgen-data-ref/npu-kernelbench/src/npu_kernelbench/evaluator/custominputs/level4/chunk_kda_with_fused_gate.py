"""Custom input generator for ChunkKdaWithFusedGate."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class ChunkKdaWithFusedGateInputGenerator(BaseInputGenerator):
    """Generates KDA with fused gate inputs."""

    names = ("ChunkKdaWithFusedGate",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name in ("q", "k"):
                out[name] = self._make_normalized(ctx, name)
            elif name == "v":
                out[name] = self._make_scaled(ctx, name, 0.2)
            elif name == "raw_g":
                out[name] = self._make_scaled(ctx, name, 0.8)
            elif name == "beta":
                out[name] = self._make_beta(ctx)
            elif name == "A_log":
                out[name] = self._make_scaled(ctx, name, 0.4)
            elif name == "g_bias":
                out[name] = self._make_scaled(ctx, name, 0.3)
            elif name == "initial_state":
                out[name] = self._make_scaled(ctx, name, 0.04)
            elif name in ("scale", "lower_bound", "use_bias", "use_lower_bound", "use_initial_state", "output_final_state", "use_qk_l2norm"):
                out[name] = self._get_scalar(ctx, name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _make_normalized(self, ctx: InputGenerationContext, name: str) -> torch.Tensor:
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        torch.manual_seed(ctx.derive_seed(name))
        t = torch.randn(shape, dtype=torch.float32, device=device)
        return F.normalize(t, p=2.0, dim=-1, eps=1e-6).to(dtype)

    def _make_scaled(self, ctx: InputGenerationContext, name: str, scale: float) -> torch.Tensor:
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        torch.manual_seed(ctx.derive_seed(name))
        t = torch.randn(shape, dtype=torch.float32, device=device) * scale
        return t.to(dtype)

    def _make_beta(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("beta")
        dtype = ctx.resolve_dtype("beta")
        device = ctx.device
        torch.manual_seed(ctx.derive_seed("beta"))
        logits = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.sigmoid(logits).to(dtype)
