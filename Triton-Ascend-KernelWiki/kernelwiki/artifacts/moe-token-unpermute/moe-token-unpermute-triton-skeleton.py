# moe_token_unpermute - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_token_unpermute/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_token_unpermute_kernel(
    permuted_tokens_ptr, sorted_indices_ptr, probs_ptr,
    unpermuted_tokens_ptr,
    num_tokens, top_k, hidden_size,
    HAVE_PROBS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # 每个 program 处理一个原始 token
    offs_j = tl.arange(0, BLOCK_H)
    j_mask = offs_j < hidden_size

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in range(top_k):
        gather_idx = pid * top_k + k
        sorted_idx = tl.load(sorted_indices_ptr + gather_idx)
        gathered = tl.load(permuted_tokens_ptr + sorted_idx * hidden_size + offs_j, mask=j_mask, other=0.0)

        if HAVE_PROBS:
            prob = tl.load(probs_ptr + gather_idx)
            gathered = gathered * prob

        acc += gathered.to(tl.float32)

    tl.store(unpermuted_tokens_ptr + pid * hidden_size + offs_j, acc, mask=j_mask)


# grid = (num_tokens,)
# AIV-only（纯 elementwise gather + mul + reduce_sum），无 Cube。
# ~16 tiling key 编码 dtype(BF16/FP16/FP32) × haveProbs(false/true) × mix mode(probs dtype ≠ token dtype)。
# 无 probs 时 topK=1，简化为纯 gather。
# 950 转调 aclnnMoeFinalizeRoutingV2。
# 与 token_permute 配对：permute 排序展开，unpermute gather+累加还原。