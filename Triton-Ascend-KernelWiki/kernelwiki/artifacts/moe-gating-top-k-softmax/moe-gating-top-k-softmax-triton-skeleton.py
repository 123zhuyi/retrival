# moe_gating_top_k_softmax - Triton-Ascend skeleton (PSEUDOCODE)
# reproducibility: pseudocode. 来源：moe/moe_gating_top_k_softmax/ (d28d8f9...)

import triton
import triton.language as tl


@triton.jit
def moe_gating_top_k_softmax_kernel(
    x_ptr, finished_ptr,
    y_ptr, expert_idx_ptr, row_idx_ptr,
    num_rows, num_experts, k,
    HAS_FINISHED: tl.constexpr,
    BLOCK_E: tl.constexpr,   # expert 块
):
    pid = tl.program_id(0)  # 每个 program 处理一行
    offs_e = tl.arange(0, BLOCK_E)
    e_mask = offs_e < num_experts

    # 阶段 1: softmax(x, axis=-1)
    x = tl.load(x_ptr + pid * num_experts + offs_e, mask=e_mask, other=-float("inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    softmax_out = x_exp / x_sum

    # 阶段 2: topK(softmax_out, k)
    # (Triton 无内置 topk，需自定义并行 topk 或多次 max+mask)
    for ki in range(k):
        val = tl.max(softmax_out, axis=0)
        idx = tl.argmax(softmax_out, axis=0)
        tl.store(y_ptr + pid * k + ki, val)
        tl.store(expert_idx_ptr + pid * k + ki, idx)
        # mask 已选位置
        softmax_out = tl.where(offs_e == idx, -float("inf"), softmax_out)

    # 阶段 3: rowIdx = arange(N*K).reshape(K,N).T
    for ki in range(k):
        tl.store(row_idx_ptr + pid * k + ki, pid)

    # 阶段 4: finished 条件填充
    if HAS_FINISHED:
        finished = tl.load(finished_ptr + pid)
        if finished:
            for ki in range(k):
                tl.store(expert_idx_ptr + pid * k + ki, num_experts)


# grid = (num_rows,)
# AIV-only，无 Cube。18 tiling key（3 模板族 × dtype）。
# E_K_FullLoad（小数据量全装载）/ K_FullLoad（中数据量 k 全装载）/ Perf（大数据量按 ColRange 优化）。
# Triton 中 topk 需自定义（Triton 无内置 topk）或 host 预排序。
# 这是 MoE pipeline 的入口算子：gating -> softmax -> topk -> init_routing -> FFN -> finalize。