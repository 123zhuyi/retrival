"""Custom input generator for IOU.

Generates legal bounding boxes for ``bboxes`` and ``gtboxes`` in
``[x1, y1, x2, y2]`` format, matching the old benchmark logic.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class IOUInputGenerator(BaseInputGenerator):
    """Generates valid bounding-box inputs for the IOU operator."""

    names = ("IOU",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        dev = ctx.device
        out: dict[str, Any] = {}

        for box_name in ("bboxes", "gtboxes"):
            if box_name not in ctx.definition.inputs:
                continue

            input_spec = ctx.workload.inputs.get(box_name)
            spec = ctx.definition.inputs[box_name]

            # Resolve shape
            if (
                isinstance(input_spec, RandomInput)
                and getattr(input_spec, "shape", None) is not None
            ):
                box_shape = tuple(input_spec.shape)
            elif spec.shape is not None:
                shapes = ctx.definition.get_input_shapes(ctx.workload.axes)
                box_shape = shapes.get(box_name)
            else:
                raise ValueError(f"IOU {box_name} requires shape, got None")

            N = box_shape[0]
            dtype = ctx.resolve_dtype(box_name)

            # Use the workload's mixed distribution for coordinates, then
            # enforce valid boxes (x2 > x1, y2 > y1) via abs widths/heights.
            coords = ctx.rand_tensor(box_name, shape=(N, 4), dtype=dtype).abs()
            xy1 = coords[:, :2]
            wh = coords[:, 2:]
            out[box_name] = torch.cat([xy1, xy1 + wh], dim=1).to(dev)

        # Build ordered output matching definition.inputs
        result: list[Any] = []
        for name in ctx.definition.inputs:
            if name in out:
                result.append(out[name])
            else:
                # mode scalar
                result.append(ctx.scalar(name))
        return result
