---
schema_version: okf.v1
kind: example
type: code_example
source_family: reference
title: Moe Dispatch Skeleton
description: 骨架：MoE Token Dispatch（Scatter + Expert GEMM） 适用：MoE 专家路由，top-k 分发
tags: [code_skeleton, moe_dispatch]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/moe_dispatch.py
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/moe_dispatch.py
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/skeletons/moe_dispatch.py` 迁移而来，原始为代码骨架，未做内容改写，仅转 OKF frontmatter。

```python
"""
骨架：MoE Token Dispatch（Scatter + Expert GEMM）
适用：MoE 专家路由，top-k 分发
经验：exp_gs_bin_003 Binned dispatch (35%), exp_gs_bin_005 work-stealing (30%)
参考：ops/gather_scatter.md, ops/fused_ops.md Fusion-6
约束：[hardware/constraints_cheatsheet]
"""

import torch
import triton
import triton.language as tl


# ────────────────────────────────────────────────────────────────
# Phase 1: Token Scatter（按 expert 分组写入 expert buffer）
# ────────────────────────────────────────────────────────────────

@triton.jit
def token_scatter_kernel(
    tokens_ptr,         # [num_tokens, hidden_dim]
    expert_buf_ptr,     # [num_experts * max_capacity, hidden_dim]（输出）
    routing_ptr,        # [num_tokens, top_k]  每个 token 的 top-k expert ID
    slot_ptr,           # [num_tokens, top_k]  每个 token 在 expert buffer 中的 slot
    num_tokens,
    hidden_dim,
    max_capacity: tl.constexpr,
    BLOCK_H: tl.constexpr,   # hidden_dim 每次处理的列数
):
    # 每个 program 处理一个 (token, top_k_slot) 对
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_id = tl.load(routing_ptr + pid)
    slot_id = tl.load(slot_ptr + pid)
    dst_row = expert_id * max_capacity + slot_id

    offs = tl.arange(0, BLOCK_H)
    for h_start in range(0, hidden_dim, BLOCK_H):
        h_offs = h_start + offs
        mask = h_offs < hidden_dim
        x = tl.load(tokens_ptr + pid * hidden_dim + h_offs, mask=mask, other=0.0)
        tl.store(expert_buf_ptr + dst_row * hidden_dim + h_offs, x, mask=mask)


# ────────────────────────────────────────────────────────────────
# Phase 2: Expert GEMM（每个 expert 独立矩阵乘）
# 注意：此处调用 ops/matmul.md 中的 matmul kernel，按 expert 批量执行
# ────────────────────────────────────────────────────────────────

@triton.jit
def expert_gemm_kernel(
    input_ptr,       # [num_experts * max_capacity, hidden_dim]
    weight_ptr,      # [num_experts, hidden_dim, ffn_dim]（各专家权重）
    output_ptr,      # [num_experts * max_capacity, ffn_dim]
    token_counts_ptr,  # [num_experts] 每个 expert 实际收到的 token 数
    max_capacity,
    hidden_dim,
    ffn_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # pid_expert: 负责哪个 expert
    # pid_tile: 在该 expert 的输出矩阵中负责哪个 (m_tile, n_tile)
    pid_expert = tl.program_id(1)
    pid_tile = tl.program_id(0)

    token_count = tl.load(token_counts_ptr + pid_expert)
    if token_count == 0:
        return  # 空 expert，跳过

    num_m = tl.cdiv(token_count, BLOCK_M)
    num_n = tl.cdiv(ffn_dim, BLOCK_N)
    m_id = pid_tile // num_n
    n_id = pid_tile % num_n

    if m_id >= num_m:
        return

    offs_m = m_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    input_base = input_ptr + pid_expert * max_capacity * hidden_dim
    weight_base = weight_ptr + pid_expert * hidden_dim * ffn_dim

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_dim, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a = tl.load(input_base + offs_m[:, None] * hidden_dim + k_offs[None, :],
                    mask=(offs_m[:, None] < token_count) & (k_offs[None, :] < hidden_dim),
                    other=0.0)
        b = tl.load(weight_base + k_offs[:, None] * ffn_dim + offs_n[None, :],
                    mask=(k_offs[:, None] < hidden_dim) & (offs_n[None, :] < ffn_dim),
                    other=0.0)
        acc += tl.dot(a, b)

    output_base = output_ptr + pid_expert * max_capacity * ffn_dim
    tl.store(output_base + offs_m[:, None] * ffn_dim + offs_n[None, :],
             acc.to(tl.float16),
             mask=(offs_m[:, None] < token_count) & (offs_n[None, :] < ffn_dim))


# ────────────────────────────────────────────────────────────────
# Phase 3: Token Gather（从 expert buffer 收集回 token 顺序）
# ────────────────────────────────────────────────────────────────

@triton.jit
def token_gather_kernel(
    expert_out_ptr,  # [num_experts * max_capacity, ffn_dim]
    output_ptr,      # [num_tokens, ffn_dim]
    routing_ptr,     # [num_tokens, top_k]
    slot_ptr,        # [num_tokens, top_k]
    num_tokens,
    ffn_dim,
    max_capacity: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    expert_id = tl.load(routing_ptr + pid)
    slot_id = tl.load(slot_ptr + pid)
    src_row = expert_id * max_capacity + slot_id

    offs = tl.arange(0, BLOCK_H)
    for h_start in range(0, ffn_dim, BLOCK_H):
        h_offs = h_start + offs
        mask = h_offs < ffn_dim
        x = tl.load(expert_out_ptr + src_row * ffn_dim + h_offs, mask=mask, other=0.0)
        tl.store(output_ptr + pid * ffn_dim + h_offs, x, mask=mask)


# ────────────────────────────────────────────────────────────────
# 顶层调度函数
# ────────────────────────────────────────────────────────────────

def moe_dispatch(
    tokens: torch.Tensor,          # [num_tokens, hidden_dim]
    expert_weights: torch.Tensor,  # [num_experts, hidden_dim, ffn_dim]
    routing: torch.Tensor,         # [num_tokens] top-1 expert ID
    max_capacity: int,             # 每个 expert 的最大 token 容量
) -> torch.Tensor:
    num_tokens, hidden_dim = tokens.shape
    num_experts, _, ffn_dim = expert_weights.shape
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)

    # 计算每个 expert 的 slot（token 在该 expert buffer 中的位置）
    slots = torch.zeros_like(routing)
    token_counts = torch.zeros(num_experts, dtype=torch.int32, device=tokens.device)
    for i in range(num_tokens):  # 实际中用 CUDA/NPU kernel 计算
        e = routing[i].item()
        if token_counts[e] < max_capacity:
            slots[i] = token_counts[e]
            token_counts[e] += 1

    # Phase 1: Scatter
    expert_buf = torch.zeros(num_experts * max_capacity, hidden_dim,
                             device=tokens.device, dtype=tokens.dtype)
    token_scatter_kernel[(num_tokens,)](
        tokens, expert_buf, routing, slots,
        num_tokens, hidden_dim, max_capacity=max_capacity, BLOCK_H=BLOCK_H,
    )

    # Phase 2: Expert GEMM
    expert_out = torch.zeros(num_experts * max_capacity, ffn_dim,
                             device=tokens.device, dtype=tokens.dtype)
    max_tiles = triton.cdiv(max_capacity, 128) * triton.cdiv(ffn_dim, 128)
    expert_gemm_kernel[(max_tiles, num_experts)](
        expert_buf, expert_weights, expert_out, token_counts,
        max_capacity, hidden_dim, ffn_dim,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=256,
    )

    # Phase 3: Gather
    output = torch.empty(num_tokens, ffn_dim, device=tokens.device, dtype=tokens.dtype)
    token_gather_kernel[(num_tokens,)](
        expert_out, output, routing, slots,
        num_tokens, ffn_dim, max_capacity=max_capacity, BLOCK_H=BLOCK_H,
    )

    return output

```

<!-- okf:related:start -->

# 相关

- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../runbooks/compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: compile time dispatch](../runbooks/compute-pipeline/exp_radix_convert_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](../runbooks/compute-pipeline/exp_solve_tril_004.md) — 同属优化知识，主题相关
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../runbooks/meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）](../runbooks/meta-skills/precompute-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](skeleton-flash_attention_causal.md) — 主题相关：共享 code_skeleton
- 相关主题: [Gemm Basic Skeleton](skeleton-gemm_basic.md) — 主题相关：共享 code_skeleton
- 相关主题: [Rmsnorm Fused Skeleton](skeleton-rmsnorm_fused.md) — 主题相关：共享 code_skeleton

<!-- okf:related:end -->
