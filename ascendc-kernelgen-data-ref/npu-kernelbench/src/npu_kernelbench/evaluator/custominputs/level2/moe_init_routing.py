"""Custom input generator for MoeInitRouting.

Reproduces the old benchmark's mixed x generation, deterministic row_idx,
and random expert_idx.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class MoeInitRoutingInputGenerator(BaseInputGenerator):
    """Generates inputs for the MoeInitRouting operator."""

    names = ("MoeInitRouting",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "x":
                out[name] = ctx.rand_tensor(name)
            elif name == "row_idx":
                out[name] = self._make_row_idx(ctx)
            elif name == "expert_idx":
                out[name] = self._make_expert_idx(ctx)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_row_idx(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Deterministic arange reshaped to (num_rows, k)."""
        shape = ctx.resolve_shape("row_idx")
        device = ctx.device
        num_rows, k = shape[0], shape[1]
        values = torch.arange(num_rows * k, dtype=torch.int32, device=device)
        return values.reshape(k, num_rows).transpose(1, 0).contiguous()

    def _make_expert_idx(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Random expert indices in [0, 8)."""
        shape = ctx.resolve_shape("expert_idx")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("expert_idx"))
        return torch.randint(0, 8, list(shape), dtype=torch.int32, generator=gen).to(device)
