# moe_token_permute - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_token_permute/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_token_permute_kernel(
    tokens_ptr, indices_ptr, permute_tokens_ptr, sorted_indices_ptr,
    num_tokens, hidden_size, top_k,
    PADDED_MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_i = pid
    offs_j = tl.arange(0, BLOCK_H)
    j_mask = offs_j < hidden_size

    if not PADDED_MODE:
        # 阶段 1: argSort(indices) -> sortedIndicesFirst
        # (Triton 无内置 argSort，需自定义并行排序或 host 预排序)
        sorted_first = tl.load(indices_ptr + offs_i)  # 简化：假设已 argSort

        # 阶段 2: argSort(sortedIndicesFirst) -> sortedIndicesOut
        sorted_out = sorted_first  # 简化
        tl.store(sorted_indices_ptr + offs_i, sorted_out)

        # 阶段 3: index_copy permuteTokens[sortedIndicesFirst[i]] = tokens[i//topK]
        src_token = tl.load(tokens_ptr + (offs_i // top_k) * hidden_size + offs_j, mask=j_mask, other=0.0)
        tl.store(permute_tokens_ptr + sorted_first * hidden_size + offs_j, src_token, mask=j_mask)
    else:
        # paddedMode: permuteTokens[i] = tokens[indices[i]]; sortedIndicesOut = indices
        idx = tl.load(indices_ptr + offs_i)
        src_token = tl.load(tokens_ptr + idx * hidden_size + offs_j, mask=j_mask, other=0.0)
        tl.store(permute_tokens_ptr + offs_i * hidden_size + offs_j, src_token, mask=j_mask)
        tl.store(sorted_indices_ptr + offs_i, idx)


# grid = (num_indices,)
# AIV-only，无 Cube。8 tiling key（1-8）= one/multicore × normal/spilt_d × padded_false/true。
# 实际仅支持 paddedMode=false（README L79），key 1-4 使用。
# 950 在框架层转调 moe_init_routing_v2，不走本 kernel。
# Triton 中 argSort 需自定义（如 bitonic sort）或 host 预排序。