# moe_init_routing - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# MoE 路由算子：对 expertIdx 和 rowIdx 排序，建立映射，gather 出 expandedX。
# 来源：moe/moe_init_routing/ (commit d28d8f9...), README L19-29.

import triton
import triton.language as tl


@triton.jit
def moe_init_routing_kernel(
    x_ptr, row_idx_ptr, expert_idx_ptr,
    expanded_x_ptr, expanded_row_idx_ptr, expanded_expert_idx_ptr,
    n, cols, k, active_num,
    FULL_LOAD: tl.constexpr,
    SMALL_ACTIVE: tl.constexpr,
):
    pid = tl.program_id(0)
    # 任务 = (expert group, token block)

    # 1. Sort：对 (expertIdx, rowIdx) 按 expert 排序
    #    得到 sorted_expert_idx, sorted_row_idx

    # 2. SrcToDst：建立映射
    #    expanded_row_idx[sorted_row_idx[i]] = i

    # 3. Gather：按 expanded_row_idx 取出 expandedX
    #    expanded_x[i] = x[sorted_row_idx[i] % n]

    # 注意：
    # - 纯 Vector kernel，无需 Cube
    # - A2（910B）/ A3（910_93）/ 950（950PR/950DT）三架构支持
    # - kernel_types: moe（非 attention）
    # - 输出：expandedX / expandedRowIdx / expandedExpertIdx
    pass