"""Custom input generator for GroupedMatmul.

When ``A`` is 2-D the old benchmark synthesises monotonically-increasing
end offsets from evenly-sized groups.  When ``A`` is 3-D (direct grouping)
``offsets`` is ``None``.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class GroupedMatmulInputGenerator(BaseInputGenerator):
    """Generates grouped-matrix-multiplication inputs."""

    names = ("GroupedMatmul",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name in ("A", "B"):
                out[name] = ctx.rand_tensor(name)
            elif name == "offsets":
                out[name] = self._make_offsets(ctx, out["A"], out["B"])
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _make_offsets(
        self,
        ctx: InputGenerationContext,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return evenly-spaced int32 end offsets when A is 2-D, else None."""
        input_spec = ctx.workload.inputs.get("offsets")

        # Explicit null scalar means "no offsets" (3-D direct grouping).
        if isinstance(input_spec, ScalarInput) and input_spec.value is None:
            return None

        if a_tensor.ndim != 2:
            return None

        total_rows = a_tensor.shape[0]
        num_groups = b_tensor.shape[0]
        if num_groups == 0:
            return None

        # Optional scalar attrs can override the derived values.
        group_size = self._scalar_value(ctx, "group_size")
        num_groups_attr = self._scalar_value(ctx, "num_groups")
        if isinstance(num_groups_attr, int):
            num_groups = num_groups_attr
        if isinstance(group_size, int):
            pass
        else:
            group_size = total_rows // num_groups if num_groups else total_rows

        offsets = torch.tensor(
            [(i + 1) * group_size for i in range(num_groups)],
            dtype=torch.int32,
            device=ctx.device,
        )
        return offsets

    def _scalar_value(self, ctx: InputGenerationContext, name: str) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return None
