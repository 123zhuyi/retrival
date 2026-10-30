"""Custom input generator for AdaptiveInstanceNormalization2DBackward.

Matches the old benchmark exactly: only ``x`` uses the legacy normal/uniform
mixed distribution; all other float tensors are drawn from a standard normal
 distribution.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class AdaptiveInstanceNormalization2DBackwardInputGenerator(BaseInputGenerator):
    """Generates inputs for the AdaptiveInstanceNormalization2DBackward operator."""

    names = ("AdaptiveInstanceNormalization2DBackward",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "x":
                out[name] = self._x_like(ctx)
            elif name == "std":
                out[name] = self._std_tensor(ctx)
            elif name in ("grad_output", "weight", "mean"):
                out[name] = self._standard_normal(ctx, name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _x_like(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Use the workload's mixed distribution for ``x``."""
        return ctx.rand_tensor("x")

    def _standard_normal(self, ctx: InputGenerationContext, name: str) -> torch.Tensor:
        """Use the workload's mixed distribution for the named tensor."""
        return ctx.rand_tensor(name)

    def _std_tensor(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Positive standard deviation in [1.0, 5.0] to avoid division by zero
        and reduce float16 overflow in ``std ** -3``.
        """
        shape = ctx.resolve_shape("std")
        dtype = ctx.resolve_dtype("std")
        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("std"))
        return (torch.rand(list(shape), generator=gen, dtype=torch.float32).to(device) * 4.0 + 1.0).to(dtype)
