"""Custom input generator for FusionAttention.

Reproduces the old benchmark's mixed-distribution float tensors for
query/key/value/pse/sink, all-ones attention masks, and scalar pass-through.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusionAttentionInputGenerator(BaseInputGenerator):
    """Generates inputs for the FusionAttention operator."""

    names = ("FusionAttention",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name in ("query", "key", "value", "pse", "sink"):
                out[name] = (
                    self._optional_tensor(ctx, name)
                    if input_spec is not None
                    else self._get_input_value(ctx, name)
                )
            elif name == "padding_mask":
                out[name] = None
            elif name == "atten_mask":
                out[name] = (
                    self._ones_mask(ctx, name)
                    if input_spec is not None
                    else self._get_input_value(ctx, name)
                )
            else:
                out[name] = self._get_input_value(ctx, name)

        return [out[name] for name in ctx.definition.inputs]

    def _optional_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Generate optional tensor if present, otherwise None."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        if name == "sink":
            return self._random_tensor(ctx, name, torch.float32)
        return self._random_tensor(ctx, name)

    def _ones_mask(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Generate an all-ones attention mask when present."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        shape = ctx.resolve_shape(name)
        dtype = ctx.resolve_dtype(name)
        return torch.ones(list(shape), dtype=dtype, device=ctx.device)

    def _random_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Independent normal/uniform branch for float tensors."""
        shape = ctx.resolve_shape(name)
        if dtype is None:
            dtype = ctx.resolve_dtype(name)
        device = ctx.device

        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed(name))
        use_normal = bool(torch.randint(0, 2, (1,), generator=gen).item() == 0)

        if use_normal:
            mu = float(torch.rand(1, generator=gen).item()) * 200.0 - 100.0
            sigma = float(torch.rand(1, generator=gen).item()) * 24.0 + 1.0
            return torch.normal(mu, sigma, list(shape), generator=gen).to(dtype).to(device)

        return (torch.rand(list(shape), generator=gen).to(device) * 10.0 - 5.0).to(dtype)
