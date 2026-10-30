"""Custom input generator for ChunkKdaBwdIntra."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


DEFAULT_CHUNK_SIZE = 64


class ChunkKdaBwdIntraInputGenerator(BaseInputGenerator):
    """Generates chunked KDA intra-chunk backward inputs."""

    names = ("ChunkKdaBwdIntra",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name in ("q", "k"):
                out[name] = self._make_normalized(ctx, name)
            elif name in ("dAqk", "dAkk", "dq", "dk", "db", "dg"):
                out[name] = ctx.rand_tensor(name)
            elif name == "beta":
                out[name] = self._make_beta(ctx)
            elif name == "g":
                out[name] = self._make_g(ctx)
            elif name == "cu_seqlens":
                out[name] = self._make_cu_seqlens(ctx)
            elif name == "chunk_size":
                out[name] = self._make_chunk_size(ctx)
            elif name in ("chunk_indices", "safe_gate"):
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
            return DEFAULT_CHUNK_SIZE
        return int(value)

    def _make_normalized(self, ctx: InputGenerationContext, name: str) -> torch.Tensor:
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        t = ctx.rand_tensor(name, shape=shape, dtype=torch.float32)
        return F.normalize(t, p=2, dim=-1).to(dtype)

    def _make_beta(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("beta")
        dtype = ctx.resolve_dtype("beta")
        device = ctx.device
        return torch.empty(shape, dtype=torch.float32, device=device).uniform_(0.0, 1.0).to(dtype)

    def _make_g(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("g")
        dtype = ctx.resolve_dtype("g")
        device = ctx.device
        B, T, HV, K = shape
        chunk_size = self._make_chunk_size(ctx)
        cu_seqlens = self._make_cu_seqlens(ctx)

        delta = -torch.empty(B, T, HV, K, device=device).uniform_(0.01, 0.2)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        g = delta.clone()
        delta_flat = delta.reshape(B * T, HV, K)
        g_flat = g.reshape(B * T, HV, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                g_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        return g.to(dtype)

    def _make_cu_seqlens(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        value = self._get_scalar(ctx, "cu_seqlens")
        if value is None:
            return None
        dtype = ctx.resolve_dtype("cu_seqlens")
        device = ctx.device
        return torch.tensor(value, dtype=dtype, device=device)
