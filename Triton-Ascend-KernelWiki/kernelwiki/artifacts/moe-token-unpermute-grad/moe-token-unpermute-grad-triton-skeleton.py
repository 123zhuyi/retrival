# moe_token_unpermute_grad - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_token_unpermute_grad/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_token_unpermute_grad_kernel(
    permuted_tokens_ptr, unpermuted_tokens_grad_ptr, sorted_indices_ptr, probs_ptr,
    permuted_tokens_grad_ptr, probs_grad_ptr,
    num_tokens, top_k, hidden_size,
    HAVE_PROBS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # 每个 program 处理一个原始 token
    offs_j = tl.arange(0, BLOCK_H)
    j_mask = offs_j < hidden_size

    grad = tl.load(unpermuted_tokens_grad_ptr + pid * hidden_size + offs_j, mask=j_mask, other=0.0)

    if HAVE_PROBS:
        probs_grad_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for k in range(top_k):
            gather_idx = pid * top_k + k
            sorted_idx = tl.load(sorted_indices_ptr + gather_idx)
            gathered = tl.load(permuted_tokens_ptr + sorted_idx * hidden_size + offs_j, mask=j_mask, other=0.0)
            prob = tl.load(probs_ptr + gather_idx)

            # probsGrad = Σ(grad * gathered)
            probs_grad_acc += grad * gathered

            # permutedTokensGrad[sortedIndices] = grad * probs
            tl.store(permuted_tokens_grad_ptr + sorted_idx * hidden_size + offs_j,
                     grad * prob, mask=j_mask)

        tl.store(probs_grad_ptr + pid * hidden_size + offs_j, probs_grad_acc, mask=j_mask)
    else:
        # 无 probs: permutedTokensGrad[sortedIndices] = unpermutedGrad
        sorted_idx = tl.load(sorted_indices_ptr + pid)
        tl.store(permuted_tokens_grad_ptr + sorted_idx * hidden_size + offs_j, grad, mask=j_mask)


# grid = (num_tokens,)
# AIV-only（scatter + 元素乘 + reduce_sum），无 Cube。
# 4 tiling key（0/1/10/11）= padded_mode(false/true) × haveProbs(false/true)。
# 2 模板：ProbNone（无 probs，纯 scatter）/ ProbNotNone（有 probs，scatter 加权 + probsGrad 求和）。
# 950 转调 aclnnMoeFinalizeRoutingV2Grad。
# 与 token_unpermute 正向配对：正向 gather+reduce，反向 scatter+reduce。