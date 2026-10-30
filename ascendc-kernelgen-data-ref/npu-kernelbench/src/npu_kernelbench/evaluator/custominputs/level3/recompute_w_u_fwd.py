"""Custom input generator for RecomputeWUFwd."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


DEFAULT_CHUNK_SIZE = 64


class RecomputeWUFwdInputGenerator(BaseInputGenerator):
    """Generates WY-representation recomputation inputs for KDA."""

    names = ("RecomputeWUFwd",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "k":
                out[name] = self._make_k(ctx)
            elif name == "v":
                out[name] = ctx.rand_tensor(name)
            elif name == "beta":
                out[name] = self._make_beta(ctx)
            elif name == "gk":
                out[name] = self._make_gk(ctx)
            elif name == "A":
                out[name] = self._make_A(ctx)
            elif name == "q":
                out[name] = self._make_q(ctx)
            elif name == "cu_seqlens":
                out[name] = self._make_cu_seqlens(ctx)
            elif name == "has_q":
                out[name] = self._get_value(ctx, "has_q")
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, CustomInput):
            return spec.value
        return None

    def _has_q(self, ctx: InputGenerationContext) -> bool:
        value = self._get_value(ctx, "has_q")
        if value is None:
            return True
        return bool(value)

    def _chunk_size(self, ctx: InputGenerationContext) -> int:
        # A.shape[-1] is the chunk size.
        return ctx.resolve_shape("A")[-1]

    def _make_k(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("k")
        dtype = ctx.resolve_dtype("k")
        device = ctx.device
        k = F.normalize(ctx.rand_tensor("k", shape=shape, dtype=torch.float32), p=2, dim=-1)
        return k.to(dtype)

    def _make_beta(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("beta")
        dtype = ctx.resolve_dtype("beta")
        device = ctx.device
        return torch.empty(shape, dtype=dtype, device=device).uniform_(0.0, 1.0)

    def _make_q(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        if not self._has_q(ctx):
            return None
        shape = ctx.resolve_shape("q")
        dtype = ctx.resolve_dtype("q")
        device = ctx.device
        q = F.normalize(ctx.rand_tensor("q", shape=shape, dtype=torch.float32), p=2, dim=-1)
        return q.to(dtype)

    def _make_gk(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("gk")
        dtype = ctx.resolve_dtype("gk")
        device = ctx.device
        B, T, HV, K = shape
        chunk_size = self._chunk_size(ctx)
        cu_seqlens = self._make_cu_seqlens(ctx)

        delta = -torch.empty(B, T, HV, K, device=device).uniform_(0.01, 0.2)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        gk = delta.clone()
        delta_flat = delta.reshape(B * T, HV, K)
        gk_flat = gk.reshape(B * T, HV, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                gk_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        return gk.to(dtype)

    def _make_A(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("A")
        dtype = ctx.resolve_dtype("A")
        device = ctx.device
        B, T, HV, BT = shape
        H = ctx.resolve_shape("k")[2]
        G = HV // H

        k = self._make_k(ctx).float()
        gk = self._make_gk(ctx).float()
        beta = self._make_beta(ctx).float()
        cu_seqlens = self._make_cu_seqlens(ctx)
        chunk_size = self._chunk_size(ctx)

        K = k.shape[-1]
        k_flat = k.reshape(B * T, H, K)
        g_flat = gk.reshape(B * T, HV, K)
        b_flat = beta.reshape(B * T, HV)
        A_flat = torch.zeros(B * T, HV, BT, dtype=torch.float32, device=device)
        head_map = torch.arange(HV, device=device) // G

        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        for (t0, t1) in bounds:
            c = t0
            while c < t1:
                e = min(c + chunk_size, t1)
                L = e - c
                k_c = k_flat[c:e][:, head_map]
                g_c = g_flat[c:e]
                b_c = b_flat[c:e]
                D = g_c.unsqueeze(1) - g_c.unsqueeze(0)
                strict = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device), -1)
                DecS = torch.where(strict[:, :, None, None], torch.exp2(D), torch.zeros(1, device=device))
                Akk_raw = torch.einsum('thd,shd,tshd->hts', k_c, k_c, DecS) * b_c.t()[:, :, None]
                eye = torch.eye(L, device=device).expand(HV, L, L)
                Akk_inv = torch.linalg.solve_triangular(eye + Akk_raw, eye, upper=False)
                A_flat[c:e, :, :L] = Akk_inv.permute(1, 0, 2)
                c = e

        return A_flat.reshape(B, T, HV, BT).to(dtype)

    def _make_cu_seqlens(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        value = self._get_value(ctx, "cu_seqlens")
        if value is None:
            return None
        dtype = ctx.resolve_dtype("cu_seqlens")
        device = ctx.device
        return torch.tensor(value, dtype=dtype, device=device)
