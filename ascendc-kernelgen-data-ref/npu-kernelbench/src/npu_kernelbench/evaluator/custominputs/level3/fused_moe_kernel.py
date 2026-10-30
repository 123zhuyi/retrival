"""Custom input generator for FusedMoeKernel."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import CustomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusedMoeKernelInputGenerator(BaseInputGenerator):
    """Generates MoE inputs with realistic activation/weight distributions."""

    names = ("FusedMoeKernel",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name == "hidden_states":
                out[name] = self._make_hidden(ctx)
            elif name == "w1":
                out[name] = self._make_w1(ctx)
            elif name == "w2":
                out[name] = self._make_w2(ctx)
            elif name == "topk_weights":
                out[name] = self._make_topk_weights(ctx)
            elif name == "topk_ids":
                out[name] = self._make_topk_ids(ctx)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_hidden(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("hidden_states")
        dtype = ctx.resolve_dtype("hidden_states")
        device = ctx.device
        # Scale standard activations to avoid fp16 overflow through SwiGLU.
        return (ctx.rand_tensor("hidden_states", shape=shape, dtype=torch.float32) * 0.2).to(dtype)

    def _make_w1(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("w1")
        dtype = ctx.resolve_dtype("w1")
        device = ctx.device
        H = shape[2]
        w = ctx.rand_tensor("w1", shape=shape, dtype=torch.float32)
        return ((w - w.mean()) * (H ** -0.5)).to(dtype)

    def _make_w2(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("w2")
        dtype = ctx.resolve_dtype("w2")
        device = ctx.device
        I = shape[2]
        w = ctx.rand_tensor("w2", shape=shape, dtype=torch.float32)
        return ((w - w.mean()) * (I ** -0.5)).to(dtype)

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
        E = ctx.resolve_shape("w1")[0]
        return torch.stack([torch.randperm(E, device=device)[:topk] for _ in range(M)]).to(dtype)
