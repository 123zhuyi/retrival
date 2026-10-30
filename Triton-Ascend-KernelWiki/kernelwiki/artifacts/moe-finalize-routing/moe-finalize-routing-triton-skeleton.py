# moe_finalize_routing - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现。reproducibility: pseudocode。
# 表达设计意图：MoE 最终路由合并（gather+scale+add，AIV-only）。
# 来源：moe/moe_finalize_routing/ (d28d8f9...), README L21-27.

import triton
import triton.language as tl


@triton.jit
def moe_finalize_routing_kernel(
    expanded_x_ptr, x1_ptr, x2_ptr, bias_ptr, scales_ptr,
    expanded_row_idx_ptr, expanded_expert_idx_ptr,
    out_ptr,
    num_rows, hidden_size, k,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_i = pid  # row
    offs_j = tl.arange(0, BLOCK_H)
    j_mask = offs_j < hidden_size

    # 跳跃连接
    out = tl.load(x1_ptr + offs_i * hidden_size + offs_j, mask=j_mask, other=0.0).to(tl.float32)
    if x2_ptr:
        out += tl.load(x2_ptr + offs_i * hidden_size + offs_j, mask=j_mask, other=0.0)

    # 遍历 K 个 expert
    for k_idx in range(k):
        # gather 索引
        row_idx = tl.load(expanded_row_idx_ptr + offs_i + k_idx * num_rows)
        expert_id = tl.load(expanded_expert_idx_ptr + offs_i + k_idx)

        # gather expandedX + bias
        expert_val = tl.load(expanded_x_ptr + row_idx * hidden_size + offs_j, mask=j_mask, other=0.0)
        bias_val = tl.load(bias_ptr + expert_id * hidden_size + offs_j, mask=j_mask, other=0.0)

        # scale
        scale = tl.load(scales_ptr + offs_i * k + k_idx)
        out += scale * (expert_val + bias_val)

    # 写回
    tl.store(out_ptr + offs_i * hidden_size + offs_j, out, mask=j_mask)


# grid = (num_rows,)
# 纯 AIV-only elementwise（gather+scale+add），无 Cube/matmul 参与。
# 19 tiling key (20000-20018) 对应 dtype(FP32/FP16/BF16)×模板变体(CutK/Cuth/K2/K4/Db/AllBias)，
#   Triton 中由 dtype constexpr + 循环策略分支处理。
# BUFFER_NUM=1 / PARALLEL_NUM=2 为 CANN 优化参数，不迁移。