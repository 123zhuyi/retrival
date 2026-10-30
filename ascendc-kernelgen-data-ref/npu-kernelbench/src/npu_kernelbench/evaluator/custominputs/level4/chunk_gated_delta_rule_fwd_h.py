"""Custom input generator for ChunkGatedDeltaRuleFwdH."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class ChunkGatedDeltaRuleFwdHInputGenerator(BaseInputGenerator):
    """Generates gated delta rule forward-H inputs."""

    names = ("ChunkGatedDeltaRuleFwdH",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "k":
                out[name] = self._make_scaled(ctx, name, 0.08)
            elif name == "w":
                out[name] = self._make_scaled(ctx, name, 0.06)
            elif name == "u":
                out[name] = self._make_scaled(ctx, name, 0.18)
            elif name in ("g", "gk"):
                out[name] = self._make_cumulative_gate(ctx, name)
            elif name == "initial_state" and name in ctx.workload.inputs:
                out[name] = self._make_scaled(ctx, name, 0.04)
            elif name == "chunk_size":
                out[name] = self._make_chunk_size(ctx)
            elif name == "gate_mode":
                out[name] = self._get_scalar(ctx, name, 0)
            elif name == "use_initial_state":
                out[name] = self._get_scalar(ctx, name, False)
            elif name == "output_final_state":
                out[name] = self._get_scalar(ctx, name, False)
            elif name == "state_v_first":
                out[name] = self._get_scalar(ctx, name, False)
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

    def _make_scaled(self, ctx: InputGenerationContext, name: str, scale: float) -> torch.Tensor:
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        torch.manual_seed(ctx.derive_seed(name))
        t = torch.randn(shape, dtype=torch.float32, device=device) * scale
        return t.to(dtype)

    def _make_cumulative_gate(self, ctx: InputGenerationContext, name: str) -> torch.Tensor:
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        T = shape[1]
        chunk_size = self._make_chunk_size(ctx)

        torch.manual_seed(ctx.derive_seed(name))
        increments = -torch.rand(shape, dtype=torch.float32, device=device) * 0.08
        output = torch.empty_like(increments)
        for start in range(0, T, chunk_size):
            end = min(T, start + chunk_size)
            output[:, start:end] = increments[:, start:end].cumsum(dim=1)
        return output.to(dtype)
