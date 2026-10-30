# moe_init_routing_quant_v2 - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_init_routing_quant_v2/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_init_routing_quant_v2_kernel(
    x_ptr, expert_idx_ptr, scale_ptr, offset_ptr,
    expanded_x_ptr, expanded_row_idx_ptr,
    expert_tokens_cumsum_ptr, dynamic_quant_scale_ptr,
    num_rows, hidden_size, expert_num, expert_capacity,
    DROP_PAD_MODE: tl.constexpr,      # 0=dropless, 1=drop
    QUANT_MODE: tl.constexpr,         # 0=static, 1=dynamic
    OUTPUT_DTYPE: tl.constexpr,       # int8 or int4
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_i = pid
    offs_j = tl.arange(0, BLOCK_H)
    j_mask = offs_j < hidden_size

    # 阶段 1: Sort (假设 host 预排序或自定义并行 sort)
    sorted_row = tl.load(expert_idx_ptr + offs_i)  # 简化
    expert_id = tl.load(expert_idx_ptr + offs_i)

    # 阶段 2: SrcToDst 反向映射
    tl.store(expanded_row_idx_ptr + sorted_row, offs_i)

    # 阶段 3: ExpertToken 统计 (dropless: cumsum, drop: histogram)
    # (Tritan 中需自定义 histogram + prefix sum)

    # 阶段 4: Quant
    x_val = tl.load(x_ptr + (sorted_row % num_rows) * hidden_size + offs_j, mask=j_mask, other=0.0)
    if QUANT_MODE == 0:  # 静态
        scale = tl.load(scale_ptr)
        offset = tl.load(offset_ptr)
        quant_result = (x_val * scale + offset).to(tl.int8)  # or int4
    else:  # 动态
        if scale_ptr:
            x_scaled = x_val * tl.load(scale_ptr)
        else:
            x_scaled = x_val
        dyn_scale = tl.max(tl.abs(x_scaled), axis=0) / 127.0
        tl.store(dynamic_quant_scale_ptr + offs_i, dyn_scale)
        quant_result = (x_scaled / dyn_scale).to(tl.int8)

    # 阶段 5: Gather + 容量处理
    if DROP_PAD_MODE == 1:  # drop
        if offs_i < expert_capacity:
            tl.store(expanded_x_ptr + offs_i * hidden_size + offs_j, quant_result, mask=j_mask)
        # 超容量丢弃，不足补零
    else:  # dropless
        tl.store(expanded_x_ptr + offs_i * hidden_size + offs_j, quant_result, mask=j_mask)


# grid = (active_num,)
# AIV-only（MIX_AIV_1_0），无 Cube。
# tiling key: 20000=静态fullload, 21000=动态fullload, 10000-11110=多阶段。
# arch35(950) 有 regbase/simt 变体，CANN 专有，不迁移。
# V2 相比 V1: 删除 rowIdx, 新增 drop/dropless + 动态quant + expertTokens 统计 + INT4 输出。