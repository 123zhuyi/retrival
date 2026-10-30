"""Custom input generator for LSTM.

Supports optional bias_ih_l0/bias_hh_l0 tensors: when the workload specifies
a scalar null value the generator emits None, matching the old benchmark.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class LSTMInputGenerator(BaseInputGenerator):
    """Generates inputs for the LSTM operator."""

    names = ("LSTM",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name in ("bias_ih_l0", "bias_hh_l0"):
                out[name] = self._optional_bias(ctx, name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _optional_bias(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Return None when the workload encodes the bias as scalar null."""
        input_spec = ctx.workload.inputs.get(name)
        if isinstance(input_spec, ScalarInput) and input_spec.value is None:
            return None
        return ctx.rand_tensor(name)
