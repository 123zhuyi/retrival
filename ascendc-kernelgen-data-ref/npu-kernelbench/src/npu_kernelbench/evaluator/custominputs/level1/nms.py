"""Custom input generator for NMS.

Generates valid bounding boxes for ``boxes`` with ``x2 > x1`` and
``y2 > y1``, matching the old benchmark logic.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class NMSInputGenerator(BaseInputGenerator):
    """Generates valid bounding-box inputs for the NMS operator."""

    names = ("NMS",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: list[Any] = []
        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name == "boxes":
                shape = ctx.resolve_shape(name)
                dtype = ctx.resolve_dtype(name)
                boxes = ctx.rand_tensor(name).to(dtype)
                x1, x2 = boxes[..., 0].clone(), boxes[..., 2].clone()
                boxes[..., 0] = torch.min(x1, x2)
                boxes[..., 2] = torch.max(x1, x2)
                y1, y2 = boxes[..., 1].clone(), boxes[..., 3].clone()
                boxes[..., 1] = torch.min(y1, y2)
                boxes[..., 3] = torch.max(y1, y2)
                out.append(boxes)
            elif isinstance(input_spec, ScalarInput):
                out.append(input_spec.value)
            elif input_spec is None and name == "pad_to_max_output_size":
                # Optional scalar not present in this workload; reference uses False.
                out.append(False)
            elif isinstance(input_spec, RandomInput):
                out.append(ctx.rand_tensor(name))
            else:
                out.append(ctx.rand_tensor(name))
        return out
