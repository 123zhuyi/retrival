"""Custom input generator for FusedMoeKernelGptqAwq."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusedMoeKernelGptqAwqInputGenerator(BaseInputGenerator):
    """Generates int4 quantized MoE weights and matching activations."""

    names = ("FusedMoeKernelGptqAwq",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "hidden_states":
                out[name] = self._make_hidden(ctx)
            elif name in ("w1_q", "w1_scale", "w1_zp"):
                out[name] = self._make_quant_weight(ctx, "w1", name)
            elif name in ("w2_q", "w2_scale", "w2_zp"):
                out[name] = self._make_quant_weight(ctx, "w2", name)
            elif name == "topk_weights":
                out[name] = self._make_topk_weights(ctx)
            elif name == "topk_ids":
                out[name] = self._make_topk_ids(ctx)
            elif isinstance(spec := ctx.workload.inputs.get(name), CustomInput) and spec.value is not None:
                out[name] = spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, CustomInput):
            return spec.value
        return None

    def _group_size(self, ctx: InputGenerationContext) -> int:
        value = self._get_value(ctx, "group_size")
        if value is None:
            return 128
        return int(value)

    def _has_zp(self, ctx: InputGenerationContext) -> int:
        value = self._get_value(ctx, "has_zp")
        if value is None:
            return 1
        return int(value)

    def _make_hidden(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("hidden_states")
        dtype = ctx.resolve_dtype("hidden_states")
        device = ctx.device
        return (ctx.rand_tensor("hidden_states", shape=shape, dtype=torch.float32) * 0.2).to(dtype)

    def _make_topk_weights(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("topk_weights")
        dtype = ctx.resolve_dtype("topk_weights")
        device = ctx.device
        return torch.softmax(torch.randn(shape, device=device), dim=-1).to(dtype)

    def _make_topk_ids(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("topk_ids")
        dtype = ctx.resolve_dtype("topk_ids")
        device = ctx.device
        M, topk = shape
        E = ctx.resolve_shape("w1_q")[0]
        return torch.stack([torch.randperm(E, device=device)[:topk] for _ in range(M)]).to(dtype)

    def _quantize_int4(
        self,
        w: torch.Tensor,
        gs: int,
        has_zp: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E, N, K = w.shape
        G = K // gs
        wg = w.permute(0, 2, 1).reshape(E, G, gs, N)
        if has_zp:
            wmin = wg.amin(dim=2, keepdim=True)
            wmax = wg.amax(dim=2, keepdim=True)
            scale = ((wmax - wmin) / 15.0).clamp_min(1e-4)
            zp = torch.round(-wmin / scale).clamp(0, 15)
            q = torch.round(wg / scale + zp).clamp(0, 15)
        else:
            wmax = wg.abs().amax(dim=2, keepdim=True)
            scale = (wmax / 7.0).clamp_min(1e-4)
            zp = torch.full_like(scale, 8.0)
            q = torch.round(wg / scale + zp).clamp(0, 15)
        q = q.reshape(E, G * gs, N).to(torch.int32)
        q_even, q_odd = q[:, 0::2], q[:, 1::2]
        w_q = (q_even | (q_odd << 4)).contiguous()
        scale = scale.reshape(E, G, N).contiguous()
        zp = zp.reshape(E, G, N).to(torch.int32)
        zp_even, zp_odd = zp[:, :, 0::2], zp[:, :, 1::2]
        w_zp = (zp_even | (zp_odd << 4)).contiguous()
        return w_q, scale, w_zp

    def _make_quant_weight(self, ctx: InputGenerationContext, base: str, name: str) -> torch.Tensor:
        device = ctx.device
        gs = self._group_size(ctx)
        has_zp = self._has_zp(ctx)

        if base == "w1":
            q_shape = ctx.resolve_shape("w1_q")
            scale_shape = ctx.resolve_shape("w1_scale")
            zp_shape = ctx.resolve_shape("w1_zp")
            scale_dtype = ctx.resolve_dtype("w1_scale")
            E, Kh2, N2 = q_shape
            H = Kh2 * 2
            w_shape = (E, N2, H)
            gs = min(gs, H)  # per-channel when group_size > K
        else:
            q_shape = ctx.resolve_shape("w2_q")
            scale_shape = ctx.resolve_shape("w2_scale")
            zp_shape = ctx.resolve_shape("w2_zp")
            scale_dtype = ctx.resolve_dtype("w2_scale")
            E, Ih2, H = q_shape
            I = Ih2 * 2
            w_shape = (E, H, I)
            gs = min(gs, I)  # per-channel when group_size > K

        w = ctx.rand_tensor(f"{base}_q", shape=w_shape, dtype=torch.float32)
        if base == "w1":
            w = (w - w.mean()) * (H ** -0.5)
        else:
            w = (w - w.mean()) * (I ** -0.5)

        w_q, scale, w_zp = self._quantize_int4(w, gs, has_zp)

        if name == f"{base}_q":
            return w_q
        if name == f"{base}_scale":
            return scale.to(scale_dtype)
        if name == f"{base}_zp":
            if has_zp:
                return w_zp
            else:
                return torch.zeros(0, dtype=torch.int32, device=device)
        raise ValueError(f"Unknown quant weight component: {name}")
