"""Custom input generator for EmbeddingWithInitialLayernormBackward.

Mirrors the old benchmark: ``input_ids`` are token indices in
[0, 10000), while all other tensors use the default mixed float
(or standard int) distribution.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class EmbeddingWithInitialLayernormBackwardInputGenerator(BaseInputGenerator):
    """Generates inputs for the EmbeddingWithInitialLayernormBackward operator."""

    names = ("EmbeddingWithInitialLayernormBackward",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "input_ids":
                out[name] = self._make_input_ids(ctx)
            elif name == "rstd":
                out[name] = self._positive_float_tensor(ctx, name)
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_input_ids(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Token indices in [0, vocab_size) matching the old benchmark."""
        shape = ctx.resolve_shape("input_ids")
        dtype = ctx.resolve_dtype("input_ids")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("input_ids"))
        return torch.randint(0, 10000, list(shape), dtype=dtype, generator=gen).to(device)

    def _positive_float_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
        low: float = 1e-3,
        high: float = 5.0,
    ) -> torch.Tensor:
        """Uniform positive float tensor, avoiding zero and negative values."""
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed(name))
        return (
            torch.rand(list(shape), generator=gen, dtype=torch.float32).to(device)
            * (high - low)
            + low
        ).to(dtype)
