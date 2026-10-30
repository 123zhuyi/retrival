"""Custom input generator for ChunkO."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class ChunkOInputGenerator(BaseInputGenerator):
    """Generates chunked GLA/KDA output-stage inputs."""

    names = ("ChunkO",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "q":
                out[name] = self._make_scaled(ctx, name, 0.1)
            elif name == "k":
                out[name] = self._make_scaled(ctx, name, 0.1)
            elif name == "v":
                out[name] = self._make_scaled(ctx, name, 0.15)
            elif name == "h":
                out[name] = self._make_scaled(ctx, name, 0.04)
            elif name == "g":
                out[name] = self._make_cumulative_gate(ctx, name)
            elif name == "g_gamma":
                out[name] = self._make_scaled(ctx, name, 0.03)
            elif name == "scale":
                out[name] = self._get_scalar(ctx, name)
            elif name == "chunk_size":
                out[name] = self._make_chunk_size(ctx)
            elif name == "gate_mode":
                out[name] = self._get_scalar(ctx, name, 0)
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
        increments = -torch.rand(shape, dtype=torch.float32, device=device) * 0.06
        output = torch.empty_like(increments)
        for start in range(0, T, chunk_size):
            end = min(T, start + chunk_size)
            output[:, start:end] = increments[:, start:end].cumsum(dim=1)
        return output.to(dtype)
