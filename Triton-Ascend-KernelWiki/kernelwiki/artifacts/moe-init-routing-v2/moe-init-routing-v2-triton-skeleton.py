# moe_init_routing_v2 - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# MoE 路由 v2：对 v1 增强，支持 expert capacity、drop/pad、token 计数。
# 来源：moe/moe_init_routing_v2/ (commit d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_init_routing_v2_kernel(
    x_ptr, expert_idx_ptr,
    expanded_x_ptr, expanded_row_idx_ptr,
    expert_tokens_count_ptr, expert_tokens_before_capacity_ptr,
    n, cols, k, active_num, expert_capacity, expert_num,
    DROP_PAD_MODE: tl.constexpr,
    COUNT_FLAG: tl.constexpr,
    BEFORE_CAPACITY_FLAG: tl.constexpr,
):
    pid = tl.program_id(0)

    # 1. sort(expertIdx) — 按 expert 排序 token
    # 2. srcToDstWithCapacity — 建立映射，受 expert_capacity 约束
    # 3. gather expandedX — 按映射取 token 特征
    # 4. expertTokenOut — 输出每 expert 的 token 数/累计和

    # 与 v1 差异：
    # - 无 rowIdx 输入
    # - 支持 expert_capacity 限制
    # - 支持 drop_pad_mode（丢弃/填充）
    # - 可选输出 expert_tokens_count/expert_tokens_before_capacity
    # - 4 架构：A2/A3/950/310P
    pass