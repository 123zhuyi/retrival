"""Custom input generator for GroupedMatmulSwigluQuant."""

from __future__ import annotations

from typing import Any

import torch

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


def _generate_non_decreasing_sequence(length: int, upper_limit: int, seed: int) -> torch.Tensor:
    """Generate a non-decreasing int64 sequence ending at ``upper_limit``.

    Matches the old benchmark implementation.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    if length == 1:
        return torch.tensor([upper_limit], dtype=torch.int64)
    random_increments = torch.randint(1, 128, (length,), generator=gen)
    sequence = torch.cumsum(random_increments, dim=0)
    scale_factor = upper_limit / sequence[-1].item()
    sequence = (sequence * scale_factor).to(torch.int64)
    sequence[-1] = upper_limit
    for i in range(length - 2, -1, -1):
        if sequence[i] > sequence[i + 1]:
            sequence[i] = sequence[i + 1]
    return sequence


class GroupedMatmulSwigluQuantInputGenerator(BaseInputGenerator):
    """Generates inputs for the GroupedMatmulSwigluQuant operator."""

    names = ("GroupedMatmulSwigluQuant",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        """Reproduce the legacy list conversion and case-index RNG sequence."""
        x_shape = ctx.resolve_shape("x")
        weight_shape = ctx.resolve_shape("weight")
        weight_scale_shape = ctx.resolve_shape("weight_scale")
        x_scale_shape = ctx.resolve_shape("x_scale")
        experts = weight_shape[0]
        gen = torch.Generator()
        gen.manual_seed(ctx.case_index + 42)

        x = torch.randint(-5, 5, x_shape, dtype=torch.int8, generator=gen)
        weight = [
            torch.randint(-5, 5, weight_shape[1:], dtype=torch.int8, generator=gen)
            for _ in range(experts)
        ]
        weight_scale = [
            torch.rand(weight_scale_shape[1:], dtype=torch.float32, generator=gen)
            * 0.049
            + 0.001
            for _ in range(experts)
        ]
        x_scale = (
            torch.rand(x_scale_shape, dtype=torch.float32, generator=gen) * 0.049
            + 0.001
        )
        group_list = _generate_non_decreasing_sequence(
            experts, x_shape[0], ctx.case_index + 42
        )

        out = {
            "x": x.to(ctx.device),
            "weight": [tensor.to(ctx.device) for tensor in weight],
            "weight_scale": [tensor.to(ctx.device) for tensor in weight_scale],
            "x_scale": x_scale.to(ctx.device),
            "group_list": group_list.to(ctx.device),
        }
        return [out[name] for name in ctx.definition.inputs]
