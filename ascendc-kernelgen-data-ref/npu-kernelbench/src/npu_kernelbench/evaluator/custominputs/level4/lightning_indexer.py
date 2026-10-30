"""Custom input generator for LightningIndexer.

Reproduces the old benchmark's mixed-distribution float tensors for
query/key/weights, arange-based block tables, and int32 sequence-length tensors.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class LightningIndexerInputGenerator(BaseInputGenerator):
    """Generates inputs for the LightningIndexer operator."""

    names = ("LightningIndexer",)
    priority = 1
    defaults = {
        "actual_seq_lengths_query": None,
        "actual_seq_lengths_key": None,
        "block_table": None,
        "layout_query": "BSND",
        "layout_key": "BSND",
        "sparse_count": 2048,
        "sparse_mode": 3,
        "pre_tokens": None,
        "next_tokens": None,
        "return_value": False,
    }

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name in ("query", "key", "weights"):
                out[name] = self._random_tensor(ctx, name)
            elif name in ("actual_seq_lengths_query", "actual_seq_lengths_key"):
                out[name] = self._seq_lengths(ctx, name)
            elif name == "block_table":
                out[name] = self._block_table(ctx, name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif name in self.defaults:
                out[name] = self.defaults[name]
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _random_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor:
        """Independent normal/uniform branch for float tensors."""
        shape = ctx.resolve_shape(name)
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

    def _seq_lengths(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Convert scalar list sequence lengths to an int32 tensor."""
        input_spec = ctx.workload.inputs.get(name)
        if not isinstance(input_spec, ScalarInput):
            return None
        value = input_spec.value
        if value is None:
            return None
        return torch.tensor(value, dtype=torch.int32, device=ctx.device)

    def _block_table(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Generate an arange-based block table when present."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        shape = ctx.resolve_shape(name)
        total_blocks = 1
        for dim in shape:
            total_blocks *= dim
        return torch.arange(total_blocks, dtype=torch.int32, device=ctx.device).reshape(shape)
