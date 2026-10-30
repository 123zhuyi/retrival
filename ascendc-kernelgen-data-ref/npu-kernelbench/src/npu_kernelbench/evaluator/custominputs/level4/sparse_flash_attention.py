"""Custom input generator for SparseFlashAttention.

Reproduces the old benchmark's mixed-distribution float tensors for
query/key/value/rope, dynamic sparse_indices range based on key layout and
sparse_block_size, arange-based block tables, and int32 sequence-length tensors.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class SparseFlashAttentionInputGenerator(BaseInputGenerator):
    """Generates inputs for the SparseFlashAttention operator."""

    names = ("SparseFlashAttention",)
    priority = 1
    defaults = {
        "block_table": None,
        "actual_seq_lengths_query": None,
        "actual_seq_lengths_kv": None,
        "query_rope": None,
        "key_rope": None,
        "sparse_block_size": 1,
        "layout_query": "BSND",
        "layout_kv": "BSND",
        "sparse_mode": 3,
        "pre_tokens": None,
        "next_tokens": None,
        "attention_mode": 0,
        "return_softmax_lse": False,
    }

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        # Generate everything except sparse_indices first so that all
        # dependencies (key, block_table, layout_kv, sparse_block_size) are
        # available when we compute the sparse_indices upper bound.
        for name in ctx.definition.inputs:
            if name == "sparse_indices":
                continue
            input_spec = ctx.workload.inputs.get(name)
            if name in ("query", "key", "value", "query_rope", "key_rope"):
                out[name] = self._optional_tensor(ctx, name)
            elif name == "block_table":
                out[name] = self._block_table(ctx, name)
            elif name in ("actual_seq_lengths_query", "actual_seq_lengths_kv"):
                out[name] = self._seq_lengths(ctx, name)
            elif isinstance(input_spec, ScalarInput):
                out[name] = input_spec.value
            elif isinstance(input_spec, RandomInput):
                out[name] = ctx.rand_tensor(name)
            elif name in self.defaults:
                out[name] = self.defaults[name]
            else:
                out[name] = None

        if "sparse_indices" in ctx.definition.inputs:
            out["sparse_indices"] = self._sparse_indices(
                ctx, out.get("key"), out.get("block_table")
            )

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
        return self._random_tensor(ctx, name)

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

    def _scalar_str(self, ctx: InputGenerationContext, name: str, default: str) -> str:
        """Return a ScalarInput string value with a fallback."""
        input_spec = ctx.workload.inputs.get(name)
        if isinstance(input_spec, ScalarInput) and isinstance(input_spec.value, str):
            return input_spec.value
        return default

    def _scalar_int(self, ctx: InputGenerationContext, name: str, default: int) -> int:
        """Return a ScalarInput int value with a fallback."""
        input_spec = ctx.workload.inputs.get(name)
        if isinstance(input_spec, ScalarInput) and isinstance(input_spec.value, int):
            return input_spec.value
        return default

    def _sparse_indices(
        self,
        ctx: InputGenerationContext,
        key: torch.Tensor | None,
        block_table: torch.Tensor | None,
    ) -> torch.Tensor:
        """Generate sparse_indices with the old benchmark's dynamic upper bound."""
        shape = ctx.resolve_shape("sparse_indices")
        layout_kv_value = self._scalar_str(ctx, "layout_kv", "BSND")
        sparse_block_size = self._scalar_int(ctx, "sparse_block_size", 1)

        if layout_kv_value == "PA_BSND" and block_table is not None:
            max_idx = block_table.shape[0] * block_table.shape[1]
        elif sparse_block_size > 1:
            key_s2 = key.shape[1] if key is not None else 1
            max_idx = max(1, key_s2 // sparse_block_size)
        else:
            key_s2 = key.shape[1] if key is not None else 1
            max_idx = max(1, key_s2)

        gen = torch.Generator(device=ctx.device)
        gen.manual_seed(ctx.derive_seed("sparse_indices"))
        return torch.randint(
            0, max_idx, list(shape), dtype=torch.int32, device=ctx.device, generator=gen
        )
