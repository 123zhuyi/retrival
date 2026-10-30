# moe_token_permute_with_ep - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_token_permute_with_ep/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_token_permute_with_ep_kernel(
    tokens_ptr, indices_ptr, probs_ptr,
    permute_tokens_ptr, sorted_indices_ptr, permute_probs_ptr,
    num_tokens, hidden_size, top_k,
    range_lo, range_hi,
    PADDED_MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_i = pid
    offs_j = tl.arange(0, BLOCK_H)
    j_mask = offs_j < hidden_size

    if not PADDED_MODE:
        # 阶段 1: argSort(indices) -> sortedIndicesFirst
        sorted_first = tl.load(indices_ptr + offs_i)  # 简化：假设已 argSort

        # 阶段 2: argSort(sortedIndicesFirst) -> sortedIndicesOut
        sorted_out = sorted_first  # 简化
        tl.store(sorted_indices_ptr + offs_i, sorted_out)

        # 阶段 3: range 切片 + tokens/probs 双输出 index_copy
        if range_lo <= sorted_out < range_hi:
            dst_idx = sorted_out - range_lo
            src_token = tl.load(tokens_ptr + (offs_i // top_k) * hidden_size + offs_j, mask=j_mask, other=0.0)
            tl.store(permute_tokens_ptr + dst_idx * hidden_size + offs_j, src_token, mask=j_mask)
            if probs_ptr:
                prob = tl.load(probs_ptr + offs_i)
                tl.store(permute_probs_ptr + dst_idx, prob)
    else:
        # paddedMode: permuteTokens[i] = tokens[indices[i]]
        idx = tl.load(indices_ptr + offs_i)
        src_token = tl.load(tokens_ptr + idx * hidden_size + offs_j, mask=j_mask, other=0.0)
        tl.store(permute_tokens_ptr + offs_i * hidden_size + offs_j, src_token, mask=j_mask)
        tl.store(sorted_indices_ptr + offs_i, idx)
        if probs_ptr:
            tl.store(permute_probs_ptr + offs_i, tl.load(probs_ptr + offs_i))


# grid = (num_indices,)
# AIV-only，无 Cube。8 tiling key（1-8）= one/multicore × normal/spilt_d × padded_false/true。
# 与 token_permute 差异：新增 probs 双输出 + range 范围切片（EP 切分）。
# range 空时框架层回退到 moe_token_permute（README L133）。
# Triton 中 argSort 需自定义（如 bitonic sort）或 host 预排序。