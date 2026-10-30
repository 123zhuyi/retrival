# moe_token_permute_with_routing_map - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：routingMap 驱动的 MoE token 置换
#   maskedSelect + argSort + index_select（两种模式：dropAndPad false/true）
# CANN 专有机制（MTE DataCopyPad / 多核归并 SyncAll / arch35 SIMD/SIMT gather）不在此体现。
# 来源：moe/moe_token_permute_with_routing_map/ (commit d28d8f9...), README.md.

import triton
import triton.language as tl


@triton.jit
def moe_token_permute_with_routing_map_kernel(
    tokens_ptr,                  # [tokens_num, H]
    routing_map_ptr,             # [tokens_num, expert_num] BOOL/INT8
    probs_ptr,                   # [tokens_num, expert_num] 可选
    permute_tokens_ptr,           # [num_out_tokens, H] 输出
    permute_probs_ptr,            # [num_out_tokens] 输出
    sorted_indices_ptr,          # [num_out_tokens] 输出
    tokens_num, expert_num, num_out_tokens, H,
    DROP_AND_PAD: tl.constexpr,
    HAS_PROBS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 1. maskedSelect: routingMap.T -> sortedIndicesFirst
    #    routing_map[t, e] = 1 表示 token t 路由到 expert e
    #    伪过程：遍历 routing_map，收集非零位置的 token 索引
    # 2. argSort: sortedIndicesFirst -> sortedIndicesOut
    #    伪过程：tl.argsort 或自定义排序

    if not DROP_AND_PAD:
        # topK 模式
        topK = num_out_tokens // tokens_num
        # 简化：假设 sorted_indices 已算出
        for i in range(num_out_tokens):
            src_idx = i // topK
            # permuteTokens[sortedIndicesOut[i]] = tokens[i // topK]
            offs_h = tl.arange(0, BLOCK_H)
            h_mask = offs_h < H
            tok = tl.load(tokens_ptr + src_idx * H + offs_h, mask=h_mask, other=0.0)
            tl.store(permute_tokens_ptr + i * H + offs_h, tok, mask=h_mask)
    else:
        # capacity 模式
        capacity = num_out_tokens // expert_num
        # sortedIndicesOut = argsort(routingMap.T, dim=-1)[:, :capacity]
        # permutedTokensOut = tokens.index_select(0, sortedIndicesOut)
        for i in range(num_out_tokens):
            # idx = sorted_indices[i]  # 伪过程
            offs_h = tl.arange(0, BLOCK_H)
            h_mask = offs_h < H
            tok = tl.load(tokens_ptr + i * H + offs_h, mask=h_mask, other=0.0)  # 简化
            tl.store(permute_tokens_ptr + i * H + offs_h, tok, mask=h_mask)

    # probs 置换（同 tokens 路径）
    if HAS_PROBS:
        pass  # 同 tokens maskedSelect + index_select


# grid = (num_out_tokens,)  # 沿输出 token 维切分
# 纯 Vector 计算（maskedSelect + sort + gather），无 Cube
# 多核排序 + 归并：tl.program_id 多核 + tl.argsort 单核 fallback