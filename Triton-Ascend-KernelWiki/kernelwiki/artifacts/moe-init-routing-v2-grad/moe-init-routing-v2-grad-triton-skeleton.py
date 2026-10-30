#!/usr/bin/env python3
# """
# MoeInitRoutingV2Grad - Triton-Ascend skeleton
# reproducibility: pseudocode
#
# 公式（README.md）：
#   gradX_i = Σ_{t=0}^{topK} gradExpandedX[expandedRowIdx[i*topK + t]]
# """
import triton
import triton.language as tl


@triton.jit
def _moe_init_routing_v2_grad_kernel(
    grad_expanded_x_ptr, expanded_row_idx_ptr, grad_x_ptr,
    H, top_k, stride_grad,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # 行块索引

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for t in range(top_k):
        idx = tl.load(expanded_row_idx_ptr + pid * top_k + t)
        gathered = tl.load(grad_expanded_x_ptr + idx * stride_grad + offs_h, mask=mask_h, other=0.0)
        acc += gathered.to(tl.float32)

    tl.store(grad_x_ptr + pid * H + offs_h, acc.to(grad_x_ptr.dtype.element_ty), mask=mask_h)