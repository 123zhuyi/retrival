#!/usr/bin/env python3
# """
# MoeGatingTopKSoftmaxV2 - Triton-Ascend skeleton
# reproducibility: pseudocode
#
# 公式（README.md）：
#   renorm=0: softmaxResult = softmax(x, axis=-1); yOut, expertIdxOut = topk(softmaxResult, k)
#   renorm=1: topkOut, expertIdxOut = topk(x, k); yOut = softmax(topkOut, axis=-1)
#   finished=True: expertIdxOut = num_expert
# """
import triton
import triton.language as tl


@triton.jit
def _moe_gating_top_k_softmax_v2_kernel(
    x_ptr, finished_ptr, y_ptr, expert_idx_ptr, softmax_result_ptr,
    num_rows, num_expert, k,
    RENORM: tl.constexpr,
    OUTPUT_SOFTMAX: tl.constexpr,
    HAS_FINISHED: tl.constexpr,
):
    pid = tl.program_id(0)  # 行索引

    offs = tl.arange(0, num_expert)
    x_row = tl.load(x_ptr + pid * num_expert + offs).to(tl.float32)

    if RENORM == 0:
        # 先 softmax 后 topk
        x_max = tl.max(x_row, axis=0)
        x_exp = tl.exp(x_row - x_max)
        x_sum = tl.sum(x_exp, axis=0)
        softmax_result = x_exp / x_sum
        if OUTPUT_SOFTMAX:
            tl.store(softmax_result_ptr + pid * num_expert + offs, softmax_result)
        topk_vals, topk_idxs = tl.topk(softmax_result, k)
    else:
        # 先 topk 后 softmax
        topk_vals, topk_idxs = tl.topk(x_row, k)
        topk_max = tl.max(topk_vals, axis=0)
        topk_exp = tl.exp(topk_vals - topk_max)
        topk_sum = tl.sum(topk_exp, axis=0)
        topk_vals = topk_exp / topk_sum

    # 写回 yOut, expertIdxOut
    k_offs = tl.arange(0, k)
    tl.store(y_ptr + pid * k + k_offs, topk_vals.to(y_ptr.dtype.element_ty))
    if HAS_FINISHED:
        finished = tl.load(finished_ptr + pid)
        topk_idxs = tl.where(finished, num_expert, topk_idxs)
    tl.store(expert_idx_ptr + pid * k + k_offs, topk_idxs.to(tl.int32))