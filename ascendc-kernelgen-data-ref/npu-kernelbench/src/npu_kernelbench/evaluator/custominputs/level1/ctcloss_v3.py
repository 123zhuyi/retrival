"""Custom input generator for CTClossV3.

Reproduces the old benchmark's log_softmax log_probs, targets that exclude
blank, and computed input/target lengths.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class CTClossV3InputGenerator(BaseInputGenerator):
    """Generates inputs for the CTCLoss operator (migrated from CTClossV3)."""

    names = ("CTCLoss",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "log_probs":
                out[name] = self._make_log_probs(ctx)
            elif name == "targets":
                out[name] = self._make_targets(ctx, out["log_probs"])
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_log_probs(self, ctx: InputGenerationContext) -> torch.Tensor:
        """Generate log_probs as log_softmax of random mixed-distribution data."""
        shape = ctx.resolve_shape("log_probs")
        dtype = ctx.resolve_dtype("log_probs")
        device = ctx.device
        is_2d = len(shape) == 2

        raw = ctx.rand_tensor("log_probs", shape=list(shape), dtype=torch.float32)
        dim = 1 if is_2d else -1
        return torch.nn.functional.log_softmax(raw, dim=dim).to(dtype)

    def _make_targets(
        self,
        ctx: InputGenerationContext,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Generate targets excluding the blank label."""
        lp_shape = log_probs.shape
        is_2d = len(lp_shape) == 2
        T = lp_shape[0]
        C = lp_shape[-1]
        N = 1 if is_2d else lp_shape[1]

        tgt_shape = ctx.resolve_shape("targets")
        S = tgt_shape[0] if len(tgt_shape) == 1 else tgt_shape[1]
        target_dtype = ctx.resolve_dtype("targets")

        blank = self._blank_value(ctx)
        labels = [i for i in range(C) if i != blank]
        if not labels:
            labels = [0]

        device = ctx.device
        gen = torch.Generator()
        gen.manual_seed(ctx.derive_seed("targets"))
        indices = torch.randint(0, len(labels), (N, S), generator=gen).to(device)
        label_tensor = torch.tensor(labels, dtype=target_dtype, device=device)
        targets = label_tensor[indices]
        if is_2d:
            return targets.squeeze(0)
        return targets

    def _blank_value(self, ctx: InputGenerationContext) -> int:
        """Return the blank scalar value, defaulting to 0."""
        spec = ctx.workload.inputs.get("blank")
        if isinstance(spec, ScalarInput):
            return int(spec.value or 0)
        return 0
