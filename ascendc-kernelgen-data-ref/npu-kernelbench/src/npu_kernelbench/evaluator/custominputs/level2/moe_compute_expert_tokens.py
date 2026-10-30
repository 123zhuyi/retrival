"""Custom input generator for MoeComputeExpertTokens.

Mirrors the old benchmark logic: ``sorted_expert_for_source_row`` is generated
as random expert indices in [0, num_expert) and then sorted in non-decreasing
order, because the operator requires a sorted expert assignment tensor.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class MoeComputeExpertTokensInputGenerator(BaseInputGenerator):
    """Generates sorted expert indices for MoeComputeExpertTokens."""

    names = ("MoeComputeExpertTokens",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "sorted_expert_for_source_row":
                out[name] = self._make_sorted_experts(ctx)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_sorted_experts(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Random expert indices in [0, num_expert), sorted ascending."""
        shape = ctx.resolve_shape("sorted_expert_for_source_row")
        num_expert = self._scalar_value(ctx, "num_expert") or 8
        device = ctx.device

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("sorted_expert_for_source_row"))
        unsorted = torch.randint(
            0, max(1, int(num_expert)), list(shape), dtype=torch.int32, generator=gen
        )
        sorted_experts, _ = torch.sort(unsorted)
        return sorted_experts.to(device)

    def _scalar_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None
