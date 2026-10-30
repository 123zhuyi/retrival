"""Custom input generator for FusedInferAttentionScore.

Reproduces the old benchmark's mixed-distribution float tensors for
query/key/value and optional floating-point parameters, all-ones attention
masks, and scalar pass-through.
"""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import RandomInput, ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


class FusedInferAttentionScoreInputGenerator(BaseInputGenerator):
    """Generates inputs for the FusedInferAttentionScore operator."""

    names = ("FusedInferAttentionScore",)
    priority = 1
    defaults = {
        "pse_shift": None,
        "atten_mask": None,
        "actual_seq_lengths": None,
        "actual_seq_lengths_kv": None,
        "dequant_scale1": None,
        "quant_scale1": None,
        "dequant_scale2": None,
        "quant_scale2": None,
        "quant_offset2": None,
        "antiquant_scale": None,
        "antiquant_offset": None,
        "block_table": None,
        "query_padding_size": None,
        "kv_padding_size": None,
        "key_antiquant_scale": None,
        "key_antiquant_offset": None,
        "value_antiquant_scale": None,
        "value_antiquant_offset": None,
        "key_shared_prefix": None,
        "value_shared_prefix": None,
        "actual_shared_prefix_len": None,
        "query_rope": None,
        "key_rope": None,
        "key_rope_antiquant_scale": None,
        "num_heads": 1,
        "scale": 1.0,
        "pre_tokens": 2147483647,
        "next_tokens": 2147483647,
        "input_layout": "BSH",
        "num_key_value_heads": 0,
        "sparse_mode": 0,
        "inner_precise": 0,
        "block_size": 0,
        "antiquant_mode": 0,
        "softmax_lse_flag": False,
        "key_antiquant_mode": 0,
        "value_antiquant_mode": 0,
    }

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        out: dict[str, Any] = {}

        for name in ctx.definition.inputs:
            input_spec = ctx.workload.inputs.get(name)
            if name in (
                "query",
                "key",
                "value",
                "pse_shift",
                "dequant_scale1",
                "quant_scale1",
                "dequant_scale2",
                "quant_scale2",
                "quant_offset2",
                "antiquant_scale",
                "antiquant_offset",
                "key_antiquant_scale",
                "key_antiquant_offset",
                "value_antiquant_scale",
                "value_antiquant_offset",
                "key_shared_prefix",
                "value_shared_prefix",
                "query_rope",
                "key_rope",
                "key_rope_antiquant_scale",
            ):
                out[name] = self._optional_tensor(ctx, name)
            elif name == "atten_mask":
                out[name] = self._ones_mask(ctx, name)
            elif name in ("actual_seq_lengths", "actual_seq_lengths_kv", "actual_shared_prefix_len"):
                out[name] = self._seq_lengths(ctx, name)
            elif name in ("query_padding_size", "kv_padding_size"):
                out[name] = self._int64_tensor(ctx, name)
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

        self._adjust_bsh_gqa_shapes(ctx, out)
        return [out[name] for name in ctx.definition.inputs]

    def _adjust_bsh_gqa_shapes(
        self,
        ctx: InputGenerationContext,
        out: dict[str, Any],
    ) -> None:
        """Apply the legacy defensive key/value reshape for BSH GQA cases."""
        num_heads = out.get("num_heads", 1)
        num_kv_heads = out.get("num_key_value_heads", 0)
        query = out.get("query")
        if (
            out.get("input_layout") != "BSH"
            or not isinstance(query, torch.Tensor)
            or not (0 < num_kv_heads < num_heads)
            or query.shape[-1] % num_heads != 0
        ):
            return

        expected_hidden = num_kv_heads * (query.shape[-1] // num_heads)
        for name in ("key", "value"):
            tensor = out.get(name)
            if isinstance(tensor, torch.Tensor) and tensor.shape[-1] != expected_hidden:
                shape = (*tensor.shape[:-1], expected_hidden)
                out[name] = ctx.rand_tensor(name, shape=shape, dtype=tensor.dtype)

    def _optional_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Generate optional tensor if present, otherwise None."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        if name in ("dequant_scale1", "quant_scale1", "dequant_scale2", "quant_scale2", "quant_offset2"):
            return self._random_tensor(ctx, name, torch.float32)
        return self._random_tensor(ctx, name)

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

    def _seq_lengths(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> list[int] | None:
        """Pass sequence-length lists through like the legacy generator."""
        input_spec = ctx.workload.inputs.get(name)
        if not isinstance(input_spec, ScalarInput):
            return None
        value = input_spec.value
        if value is None:
            return None
        return value

    def _int64_tensor(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Generate int64 tensor from scalar list when present."""
        input_spec = ctx.workload.inputs.get(name)
        if not isinstance(input_spec, ScalarInput):
            return None
        value = input_spec.value
        if value is None:
            return None
        return torch.tensor(value, dtype=torch.int64, device=ctx.device)

    def _block_table(
        self,
        ctx: InputGenerationContext,
        name: str,
    ) -> torch.Tensor | None:
        """Generate an arange-based block table when present."""
        input_spec = ctx.workload.inputs.get(name)
        if input_spec is None:
            return None
        if isinstance(input_spec, ScalarInput):
            if input_spec.value is None:
                return None
            return torch.tensor(
                input_spec.value, dtype=torch.int32, device=ctx.device
            )
        shape = ctx.resolve_shape(name)
        total_blocks = 1
        for dim in shape:
            total_blocks *= dim
        return torch.arange(total_blocks, dtype=torch.int32, device=ctx.device).reshape(shape)
