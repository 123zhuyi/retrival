"""Custom input generator for ChunkGlaFwdKernelO."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


DEFAULT_CHUNK_SIZE = 64


class ChunkGlaFwdKernelOInputGenerator(BaseInputGenerator):
    """Generates chunked GLA output-stage inputs."""

    names = ("ChunkGlaFwdKernelO",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "q":
                out[name] = self._make_q(ctx)
            elif name in ("v", "h"):
                out[name] = ctx.rand_tensor(name)
            elif name == "A":
                out[name] = self._make_A(ctx)
            elif name == "o":
                out[name] = self._make_o(ctx)
            elif name == "g":
                out[name] = self._make_g(ctx)
            elif name == "scale":
                out[name] = self._get_value(ctx, "scale")
            elif name == "cu_seqlens":
                out[name] = self._make_cu_seqlens(ctx)
            elif name == "chunk_size":
                out[name] = self._make_chunk_size(ctx)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, CustomInput):
            return spec.value
        return None

    def _make_q(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("q")
        dtype = ctx.resolve_dtype("q")
        device = ctx.device
        q = F.normalize(ctx.rand_tensor("q", shape=shape, dtype=torch.float32), p=2, dim=-1)
        return q.to(dtype)

    def _make_A(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("A")
        dtype = ctx.resolve_dtype("A")
        device = ctx.device
        scale = self._get_value(ctx, "scale")
        if scale is None:
            scale = 1.0
        return ctx.rand_tensor("A", shape=shape, dtype=dtype) * scale

    def _make_o(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("o")
        dtype = ctx.resolve_dtype("o")
        device = ctx.device
        return torch.zeros(shape, dtype=dtype, device=device)

    def _make_g(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("g")
        dtype = ctx.resolve_dtype("g")
        device = ctx.device
        B, T, H, K = shape
        chunk_size = self._make_chunk_size(ctx)
        cu_seqlens = self._make_cu_seqlens(ctx)

        delta = -torch.empty(B, T, H, K, device=device).uniform_(0.01, 0.15)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        g = delta.clone()
        delta_flat = delta.reshape(B * T, H, K)
        g_flat = g.reshape(B * T, H, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                g_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        return g.to(dtype)

    def _make_cu_seqlens(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        value = self._get_value(ctx, "cu_seqlens")
        if value is None:
            return None
        dtype = ctx.resolve_dtype("cu_seqlens")
        device = ctx.device
        return torch.tensor(value, dtype=dtype, device=device)

    def _make_chunk_size(self, ctx: InputGenerationContext) -> int:
        value = self._get_value(ctx, "chunk_size")
        if value is None:
            return DEFAULT_CHUNK_SIZE
        return int(value)
