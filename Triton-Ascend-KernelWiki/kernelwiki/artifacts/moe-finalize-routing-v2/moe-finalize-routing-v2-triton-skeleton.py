#!/usr/bin/env python3
# """
# MoeFinalizeRoutingV2 - Triton-Ascend skeleton
# reproducibility: pseudocode
#
# 公式（README.md L22-27）：
#   out(i,j) = x1(i,j) + x2(i,j) + Σ_k (scales(i,k) * (expandedX[expandedRowIdx(i+k*num_rows), j] + bias(expertIdx[i,k], j)))
# """
import triton
import triton.language as tl


@triton.jit
def _moe_finalize_routing_v2_kernel(
    expanded_x_ptr, expanded_row_idx_ptr, x1_ptr, x2_ptr, bias_ptr, scales_ptr, expert_idx_ptr, out_ptr,
    H, K, num_rows, stride_expanded_x,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # 行块索引

    offs_h = pid * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # 基础值 x1 + x2
    out_row = tl.zeros([BLOCK_H], dtype=tl.float32)
    if x1_ptr:
        out_row += tl.load(x1_ptr + offs_h, mask=mask_h, other=0.0)
    if x2_ptr:
        out_row += tl.load(x2_ptr + offs_h, mask=mask_h, other=0.0)

    # 跨 K 累加
    for k in range(K):
        idx = tl.load(expanded_row_idx_ptr + pid * K + k)  # expandedRowIdx(i + k*num_rows)
        expert = tl.load(expert_idx_ptr + pid * K + k) if expert_idx_ptr else 0
        gathered = tl.load(expanded_x_ptr + idx * stride_expanded_x + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + expert * H + offs_h, mask=mask_h, other=0.0).to(tl.float32) if bias_ptr else 0.0
        scale = tl.load(scales_ptr + pid * K + k) if scales_ptr else 1.0
        out_row += scale * (gathered + bias)

    tl.store(out_ptr + pid * H + offs_h, out_row.to(out_ptr.dtype.element_ty), mask=mask_h)