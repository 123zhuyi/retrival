# moe_compute_expert_tokens - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_compute_expert_tokens/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_compute_expert_tokens_kernel(
    sorted_experts_ptr, out_ptr,
    num_rows, num_experts,
    BLOCK_N: tl.constexpr,
):
    # 每个 program 处理一个专家的二分查找
    pid = tl.program_id(0)  # expert id
    expert_id = pid

    # 二分查找：在 sorted_experts 中找 expert_id 最后出现的行位置
    lo = 0
    hi = num_rows
    while lo < hi:
        mid = (lo + hi) // 2
        val = tl.load(sorted_experts_ptr + mid)
        if val <= expert_id:
            lo = mid + 1
        else:
            hi = mid
    # lo = expert_id 最后出现的行 + 1 = 累积 token 数
    tl.store(out_ptr + expert_id, lo)


# grid = (num_experts,)
# AIV-only（MIX_AIV_1_0），无 Cube。
# 3 tiling key（1001/1002/1003）= SS/M/L 按数据量选择。
# sortedExperts 已排序（来自 init_routing 排序输出）。
# Triton 中二分查找需用 while 循环（串行）或自定义并行实现。
# 这是 MoE pipeline 的辅助算子：统计每个专家处理的 token 数量。