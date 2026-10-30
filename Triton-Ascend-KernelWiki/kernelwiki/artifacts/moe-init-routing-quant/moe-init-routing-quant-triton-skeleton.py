# moe_init_routing_quant - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_init_routing_quant/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_init_routing_quant_kernel(
    x_ptr, row_idx_ptr, expert_idx_ptr,
    expanded_x_ptr, expanded_row_idx_ptr, expanded_expert_idx_ptr,
    num_rows, hidden_size,
    scale, offset,
    BLOCK_H: tl.constexpr,
):
    # 阶段 1: Sort by expertIdx -> expandedExpertIdx + sortedRowIdx
    # (Triton 无内置 sort，需自定义并行排序或 host 预排序)
    pid = tl.program_id(0)
    offs_i = pid
    offs_j = tl.arange(0, BLOCK_H)
    j_mask = offs_j < hidden_size

    expert_id = tl.load(expert_idx_ptr + offs_i)
    sorted_row = tl.load(row_idx_ptr + offs_i)  # 简化：假设已排序
    tl.store(expanded_expert_idx_ptr + offs_i, expert_id)

    # 阶段 2: SrcToDst 反向映射 expandedRowIdx[sortedRowIdx[i]] = i
    tl.store(expanded_row_idx_ptr + sorted_row, offs_i)

    # 阶段 3: Gather + Quant
    x_val = tl.load(x_ptr + (sorted_row % num_rows) * hidden_size + offs_j, mask=j_mask, other=0.0)
    quantized = (x_val * scale + offset).to(tl.int8)
    tl.store(expanded_x_ptr + offs_i * hidden_size + offs_j, quantized, mask=j_mask)


# grid = (active_num,)
# AIV-only，无 Cube。4 tiling key（1/2 normal + 3/4 small_active_row × 1core/multicore）。
# Triton 中 sort 需自定义（如 bitonic sort）或 host 预排序后传入。