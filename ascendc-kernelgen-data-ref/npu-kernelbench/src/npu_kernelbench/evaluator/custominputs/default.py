"""Default input generator — handles generic InputSpec types."""

from __future__ import annotations

import ast
from functools import lru_cache
from typing import Any

from npu_kernelbench.data.dtypes import dtype_str_to_torch_dtype
from npu_kernelbench.data.workload import (
    CustomInput,
    RandomInput,
    SafetensorsInput,
    ScalarInput,
    TensorListInput,
)

from .base import BaseInputGenerator
from .context import InputGenerationContext


@lru_cache(maxsize=256)
def _optional_run_params(reference_code: str) -> frozenset[str]:
    """Return the names of ``run()`` parameters that carry a default value.

    A definition input that is absent from a workload is treated as optional
    (and emitted as ``None``) only when the reference ``run`` signature gives
    that parameter a default (e.g. ``weight=None``, ``smooth_scales=None``).
    An absent input with no reference default is considered required and raises.
    """
    try:
        module = ast.parse(reference_code, mode="exec")
    except SyntaxError:
        return frozenset()
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            args = node.args
            names: set[str] = set()
            if args.defaults:
                names.update(a.arg for a in args.args[-len(args.defaults):])
            for kwonly, default in zip(args.kwonlyargs, args.kw_defaults):
                if default is not None:
                    names.add(kwonly.arg)
            return frozenset(names)
    return frozenset()


class DefaultInputGenerator(BaseInputGenerator):
    """Catch-all generator for workloads without a custom generator.

    Iterates ``definition.inputs`` in order and dispatches based on the
    matching workload ``InputSpec`` type. Contains no operator-specific logic.
    """

    priority = -1

    def can_generate(self, definition) -> bool:
        """Default generator handles every definition."""
        return True

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        optional = _optional_run_params(ctx.definition.reference)
        out: list[Any] = []
        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)

            if input_spec is None:
                # Definition declares this input but the workload omits it.
                # Optional inputs (reference run() has a default, e.g. weight=None)
                # are emitted as None so the reference/solution default applies.
                # Required inputs must be present in the workload.
                if name in optional:
                    out.append(None)
                    continue
                raise ValueError(
                    f"Required input '{name}' is absent from workload "
                    f"'{ctx.workload.uuid}' and has no default in reference run()."
                )

            if isinstance(input_spec, SafetensorsInput):
                out.append(ctx.safetensor(name))

            elif isinstance(input_spec, TensorListInput):
                dtype = dtype_str_to_torch_dtype(input_spec.dtype)
                tensors = [
                    ctx.rand_tensor(name, shape=tuple(s), dtype=dtype)
                    for s in input_spec.shapes
                ]
                out.append(tensors)

            elif isinstance(input_spec, ScalarInput):
                out.append(input_spec.value)

            elif isinstance(input_spec, CustomInput):
                out.append(ctx.custom_tensor(name))

            else:
                # RandomInput or fallback: resolve shape and generate tensor
                out.append(ctx.rand_tensor(name))

        return out
