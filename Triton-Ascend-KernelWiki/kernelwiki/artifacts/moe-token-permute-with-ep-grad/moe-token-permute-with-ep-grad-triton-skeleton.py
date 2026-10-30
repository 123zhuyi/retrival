#!/usr/bin/env python3
# """
# MoeTokenPermuteWithEpGrad - Triton-Ascend skeleton
# reproducibility: pseudocode
#
# 公式（README.md）：
#   1. sortedIndices = sortedIndices[range[0] <= i < range[1]]
#   2. tokenGradOut = permutedTokensOutputGrad.indexSelect(0, sortedIndices)
#   3. tokenGradOut = tokenGradOut.reshape(-1, topK, hiddenSize)
#   4. tokenGradOut = tokenGradOut.sum(dim=1)
# """
import triton
import triton.language as tl


@triton.jit
def _moe_token_permute_with_ep_grad_kernel(
    permuted_tokens_output_grad_ptr, sorted_indices_ptr, permuted_probs_output_grad_ptr,
    token_grad_ptr, probs_grad_ptr,
    num_tokens, top_k, hidden_size, range_start, range_end,
    BLOCK_H: tl.constexpr,
    HAS_PROBS: tl.constexpr,
):
    pid = tl.program_id(0)  # token 索引

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_size

    # 跨 topK 累加
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in range(top_k):
        # sortedIndices 索引（range 过滤后）
        idx = tl.load(sorted_indices_ptr + pid * top_k + k)
        gathered = tl.load(permuted_tokens_output_grad_ptr + idx * hidden_size + offs_h, mask=mask_h, other=0.0)
        acc += gathered.to(tl.float32)

    tl.store(token_grad_ptr + pid * hidden_size + offs_h, acc.to(token_grad_ptr.dtype.element_ty), mask=mask_h)

    # probs 梯度（若有）
    if HAS_PROBS:
        acc_probs = tl.zeros([1], dtype=tl.float32)
        for k in range(top_k):
            idx = tl.load(sorted_indices_ptr + pid * top_k + k)
            gathered_probs = tl.load(permuted_probs_output_grad_ptr + idx)
            acc_probs += gathered_probs.to(tl.float32)
        tl.store(probs_grad_ptr + pid, acc_probs.to(probs_grad_ptr.dtype.element_ty))