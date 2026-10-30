"""Custom input generator for TopKTopP.

Reproduces the old benchmark's clamped k and rand-generated p. Logits use
the workload's mixed distribution parameters (mu/sigma and uniform range).
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class TopKTopPInputGenerator(BaseInputGenerator):
    """Generates TopKTopP inputs with legal k/p ranges."""

    names = ("TopKTopP",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "logits":
                out[name] = ctx.rand_tensor(name, distribution="mixed")
            elif name == "p":
                out[name] = self._p(ctx)
            elif name == "k":
                out[name] = self._k(ctx)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _p(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("p")
        dtype = ctx.resolve_dtype("p")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("p"))
        return torch.rand(list(shape), dtype=dtype, generator=gen).to(device)

    def _k(self, ctx: InputGenerationContext) -> torch.Tensor:
        shape = ctx.resolve_shape("k")
        dtype = ctx.resolve_dtype("k")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("k"))
        k = torch.randint(0, 1000, list(shape), dtype=dtype, generator=gen)
        logits_shape = ctx.resolve_shape("logits")
        N = logits_shape[-1]
        return k.clamp(min=1, max=min(1024, N)).to(device)
