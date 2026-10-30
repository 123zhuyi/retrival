# moe_token_permute_grad - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 表达设计意图：MoE token 置换反向（unpermute = indexSelect + reshape + sum）
# CANN 专有机制（MTE DataCopyPad / 950 代理转调用）不在此体现。
# 来源：moe/moe_token_permute_grad/ (commit d28d8f9...), README.md.

import triton
import triton.language as tl


@triton.jit
def moe_token_permute_grad_kernel(
    permuted_output_grad_ptr,    # [R * K, H] 置换后的梯度
    sorted_indices_ptr,          # [R * K] 排序索引
    input_grad_ptr,              # [R, H] 输出（原始 token 梯度）
    R, K, H,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)  # 沿 R 轴切分

    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H

    # 对每个原始 token，gather 其 numTopk 个置换位置的梯度并求和
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in range(K):
        idx = tl.load(sorted_indices_ptr + pid * K + k)  # sortedIndices[row*K + k]
        grad = tl.load(permuted_output_grad_ptr + idx * H + offs_h,
                       mask=h_mask, other=0.0).to(tl.float32)
        acc += grad  # sum(dim=1)

    tl.store(input_grad_ptr + pid * H + offs_h,
             acc.to(input_grad_ptr.dtype.element_ty()), mask=h_mask)


# grid = (R,)  # 沿 R 轴切分
# 纯 Vector 计算（indexSelect + sum），无 Cube
# 950 架构：框架转调用 MoeInitRoutingV2Grad（不在此体现）