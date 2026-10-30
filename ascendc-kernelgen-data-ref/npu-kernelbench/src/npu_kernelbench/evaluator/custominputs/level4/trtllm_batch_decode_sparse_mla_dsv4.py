"""Custom input generator for TrtllmBatchDecodeSparseMlaDsv4."""

from __future__ import annotations

from typing import Any

import torch

from npu_kernelbench.data.workload import ScalarInput

from ..base import BaseInputGenerator
from ..context import InputGenerationContext


DSV4_HEAD_DIM = 512
DSV4_SWA_TOPK = 128


class TrtllmBatchDecodeSparseMlaDsv4InputGenerator(BaseInputGenerator):
    """Generates sparse MLA DSv4 decode indices."""

    names = ("TrtllmBatchDecodeSparseMlaDsv4",)
    priority = 1

    def generate(self, ctx: InputGenerationContext) -> list[Any]:
        torch.manual_seed(ctx.derive_seed())

        out: dict[str, Any] = {}
        for name in ctx.definition.inputs:
            if name in ("query", "swa_kv_cache", "compressed_kv_cache"):
                out[name] = ctx.rand_tensor(name)
            elif name == "sparse_indices":
                out[name] = self._make_sparse_indices(ctx)
            elif name == "sparse_topk_lens":
                out[name] = self._make_sparse_topk_lens(ctx)
            elif name == "seq_lens":
                out[name] = self._make_seq_lens(ctx)
            elif name == "sinks":
                out[name] = self._make_sinks(ctx)
            elif name == "cum_seq_lens_q":
                out[name] = self._make_cum_seq_lens_q(ctx)
            elif name in ("bmm1_scale", "bmm2_scale"):
                out[name] = self._get_scalar(ctx, name)
            else:
                out[name] = None

        return [out[name] for name in ctx.definition.inputs]

    def _get_scalar(self, ctx: InputGenerationContext, name: str, default: Any = None) -> Any:
        spec = ctx.workload.inputs.get(name)
        if isinstance(spec, ScalarInput):
            return spec.value
        return default

    def _kv_lens(self, ctx: InputGenerationContext) -> list[int]:
        return list(self._get_scalar(ctx, "kv_lens", []))

    def _qo_lens(self, ctx: InputGenerationContext) -> list[int] | None:
        value = self._get_scalar(ctx, "qo_lens", None)
        if value is None:
            return None
        return list(value)

    def _make_cum_seq_lens_q(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        qo_lens = self._qo_lens(ctx)
        if qo_lens is None:
            return None
        device = ctx.device
        indptr = [0]
        for l in qo_lens:
            indptr.append(indptr[-1] + l)
        return torch.tensor(indptr, dtype=torch.int32, device=device)

    def _make_block_table(self, npages: list[int], total_pages: int, device: torch.device) -> torch.Tensor:
        max_pages = max(npages)
        perm = torch.randperm(total_pages, dtype=torch.int64, device=device)
        bt = torch.zeros(len(npages), max_pages, dtype=torch.int64, device=device)
        off = 0
        for b, n in enumerate(npages):
            bt[b, :n] = perm[off:off + n]
            off += n
        return bt

    def _abs_to_flat(self, abs_idx: torch.Tensor, bt_row: torch.Tensor, ps: int) -> torch.Tensor:
        safe = abs_idx.clamp_min(0)
        flat = bt_row[safe // ps] * ps + (safe % ps)
        flat[abs_idx < 0] = -1
        return flat

    def _make_sparse_indices(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        qo_lens = self._qo_lens(ctx)
        swa_ps = int(self._get_scalar(ctx, "swa_page_size", 0))
        comp_ps = int(self._get_scalar(ctx, "compressed_page_size", 0))
        comp_ratio = int(self._get_scalar(ctx, "compressed_ratio", 0))
        comp_topk = int(self._get_scalar(ctx, "compressed_topk", 0))
        device = ctx.device

        q_shape = ctx.resolve_shape("query")
        if len(q_shape) == 4:
            sum_q = q_shape[0] * q_shape[1]
        else:
            sum_q = q_shape[0]
        B = len(kv_lens)
        H = q_shape[-2]
        capacity = DSV4_SWA_TOPK + comp_topk

        if qo_lens is None:
            q_len = sum_q // B
            qo_lens = [q_len] * B
            cum = None
        else:
            cum = [0] + [sum(qo_lens[:i + 1]) for i in range(B)]

        swa_npages = [(l + swa_ps - 1) // swa_ps for l in kv_lens]
        swa_total = ctx.resolve_shape("swa_kv_cache")[0]
        swa_bt = self._make_block_table(swa_npages, swa_total, device)

        c_lens = [(l + comp_ratio - 1) // comp_ratio for l in kv_lens] if comp_topk > 0 else [0] * B
        comp_npages = [max((l + comp_ps - 1) // comp_ps, 0) for l in c_lens]
        if comp_topk > 0:
            comp_total = ctx.resolve_shape("compressed_kv_cache")[0]
            comp_bt = self._make_block_table(comp_npages, comp_total, device)
        else:
            comp_bt = None

        sparse_indices = torch.full((sum_q, capacity), -1, dtype=torch.int32, device=device)
        for b in range(B):
            kv_len, q_len_b = kv_lens[b], qo_lens[b]
            for t in range(q_len_b):
                j = (b * qo_lens[0] + t) if cum is None else (cum[b] + t)
                token_idx = kv_len - q_len_b + t
                n_swa = min(DSV4_SWA_TOPK, token_idx + 1)
                abs_swa = torch.arange(token_idx - n_swa + 1, token_idx + 1, dtype=torch.int64, device=device)
                flat_swa = self._abs_to_flat(abs_swa, swa_bt[b], swa_ps)
                sparse_indices[j, :DSV4_SWA_TOPK] = torch.cat([
                    flat_swa.int(),
                    torch.full((DSV4_SWA_TOPK - n_swa,), -1, dtype=torch.int32, device=device),
                ])
                if comp_topk > 0:
                    base = min(comp_topk, c_lens[b])
                    n_comp = max(base - (b + t) % 4, 0)
                    if n_comp > 0:
                        abs_comp = torch.randperm(c_lens[b], device=device)[:n_comp]
                        flat_comp = self._abs_to_flat(abs_comp, comp_bt[b], comp_ps)
                        sparse_indices[j, DSV4_SWA_TOPK:DSV4_SWA_TOPK + n_comp] = flat_comp.int()
        return sparse_indices

    def _make_sparse_topk_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        qo_lens = self._qo_lens(ctx)
        comp_topk = int(self._get_scalar(ctx, "compressed_topk", 0))
        comp_ratio = int(self._get_scalar(ctx, "compressed_ratio", 0))
        device = ctx.device

        q_shape = ctx.resolve_shape("query")
        if len(q_shape) == 4:
            sum_q = q_shape[0] * q_shape[1]
        else:
            sum_q = q_shape[0]
        B = len(kv_lens)

        if qo_lens is None:
            q_len = sum_q // B
            qo_lens = [q_len] * B
            cum = None
        else:
            cum = [0] + [sum(qo_lens[:i + 1]) for i in range(B)]

        c_lens = [(l + comp_ratio - 1) // comp_ratio for l in kv_lens] if comp_topk > 0 else [0] * B
        sparse_topk_lens = torch.full((sum_q,), DSV4_SWA_TOPK, dtype=torch.int32, device=device)
        for b in range(B):
            kv_len, q_len_b = kv_lens[b], qo_lens[b]
            for t in range(q_len_b):
                j = (b * qo_lens[0] + t) if cum is None else (cum[b] + t)
                if comp_topk > 0:
                    base = min(comp_topk, c_lens[b])
                    n_comp = max(base - (b + t) % 4, 0)
                    sparse_topk_lens[j] = DSV4_SWA_TOPK + n_comp
        return sparse_topk_lens

    def _make_seq_lens(self, ctx: InputGenerationContext) -> torch.Tensor:
        kv_lens = self._kv_lens(ctx)
        device = ctx.device
        return torch.tensor(kv_lens, dtype=torch.int32, device=device)

    def _make_sinks(self, ctx: InputGenerationContext) -> torch.Tensor | None:
        has_sinks = bool(self._get_scalar(ctx, "has_sinks", False))
        if not has_sinks:
            return None
        H = ctx.resolve_shape("query")[-2]
        device = ctx.device
        return torch.randn(H, dtype=torch.float32, device=device)
