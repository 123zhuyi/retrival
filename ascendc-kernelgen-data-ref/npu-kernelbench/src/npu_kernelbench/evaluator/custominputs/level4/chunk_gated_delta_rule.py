"""Custom input generator for ChunkGatedDeltaRule."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class ChunkGatedDeltaRuleInputGenerator(BaseInputGenerator):
    """Generates chunked gated delta rule inputs."""

    names = ("ChunkGatedDeltaRule",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        raw_gate = bool(self._get_scalar(ctx, "use_gate_in_kernel", True))
        raw_beta = bool(self._get_scalar(ctx, "use_beta_sigmoid", True))

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name in ("q", "k"):
                out[name] = self._make_normalized(ctx, name)
            elif name == "v":
                out[name] = self._make_scaled(ctx, name, 0.2)
            elif name == "g":
                out[name] = self._make_g(ctx, raw_gate)
            elif name == "beta":
                out[name] = self._make_beta(ctx, raw_beta)
            elif name == "A_log":
                out[name] = self._make_scaled(ctx, name, 0.4)
            elif name == "dt_bias":
                out[name] = self._make_scaled(ctx, name, 0.3)
            elif name == "initial_state":
                out[name] = self._make_scaled(ctx, name, 0.04)
            elif name == "scale":
                out[name] = self._get_scalar(ctx, name)
            elif name == "chunk_size":
                out[name] = self._make_chunk_size(ctx)
            elif name in ("use_gate_in_kernel", "use_bias", "use_initial_state", "output_final_state", "use_qk_l2norm", "use_beta_sigmoid", "allow_neg_eigval", "state_v_first"):
                out[name] = self._get_scalar(ctx, name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _make_chunk_size(self, ctx: InputGenerationContext) -> int:
        value = self._get_scalar(ctx, "chunk_size")
        if value is None:
            raise ValueError("chunk_size is required")
        return int(value)

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

    def _make_g(self, ctx: InputGenerationContext, raw_gate: bool) -> torch.Tensor:
        shape = ctx.resolve_shape("g")
        dtype = ctx.resolve_dtype("g")
        device = ctx.device
        torch.manual_seed(ctx.derive_seed("g"))
        t = torch.randn(shape, dtype=torch.float32, device=device)
        if raw_gate:
            t = t * 0.8
        else:
            t = -F.softplus(t) * 0.08
        return t.to(dtype)

    def _make_beta(self, ctx: InputGenerationContext, raw_beta: bool) -> torch.Tensor:
        shape = ctx.resolve_shape("beta")
        dtype = ctx.resolve_dtype("beta")
        device = ctx.device
        torch.manual_seed(ctx.derive_seed("beta"))
        t = torch.randn(shape, dtype=torch.float32, device=device)
        if not raw_beta:
            t = torch.sigmoid(t)
        return t.to(dtype)
