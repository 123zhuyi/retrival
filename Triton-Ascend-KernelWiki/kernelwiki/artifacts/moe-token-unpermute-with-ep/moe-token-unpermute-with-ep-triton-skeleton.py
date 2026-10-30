#!/usr/bin/env python3
# """
# MoeTokenUnpermuteWithEp - Triton-Ascend skeleton
# reproducibility: pseudocode
#
# 公式（README.md）：
#   1. sortedIndices = sortedIndices[range[0] <= i < range[1]]
#   2. permutedTokens = permutedTokens.indexSelect(0, sortedIndices)
#   3. 若有 probs: permutedTokens_k *= probs[i,j]
#   4. out_i = Σ_{k=i*numTopk}^{(i+1)*numTopk-1} permutedTokens_k
# """
import triton
import triton.language as tl


@triton.jit
def _moe_token_unpermute_with_ep_kernel(
    permuted_tokens_ptr, sorted_indices_ptr, probs_ptr, unpermuted_tokens_ptr,
    num_tokens, top_k, hidden_size, range_start, range_end,
    BLOCK_H: tl.constexpr,
    HAS_PROBS: tl.constexpr,
):
    pid = tl.program_id(0)  # token 索引

    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_size

    # 跨 topK 累加
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for j in range(top_k):
        idx = tl.load(sorted_indices_ptr + pid * top_k + j)
        gathered = tl.load(permuted_tokens_ptr + idx * hidden_size + offs_h, mask=mask_h, other=0.0)
        if HAS_PROBS:
            prob = tl.load(probs_ptr + pid * top_k + j)
            gathered = gathered * prob
        acc += gathered.to(tl.float32)

    tl.store(unpermuted_tokens_ptr + pid * hidden_size + offs_h, acc.to(unpermuted_tokens_ptr.dtype.element_ty), mask=mask_h)