---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: three tier hierarchical dispatch'
description: 'solve_tril 场景下，Scheduling Optimization: three tier hierarchical dispatch，收益 通过分治将 O(BT³) 求逆降为 O(BT²) 的递推 + O(BT) 个 16×16 矩阵乘法'
tags: [多核并行与负载均衡, linear-algebra-operations, solve_tril]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: three tier hierarchical dispatch

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：仅支持 BT ∈ {16,32,64}，其他块大小需新增层级；16×16 原子粒度由 NPU 矩阵单元决定，换硬件需重选；分治 merge 层数随 BT 增长，BT=64 需 3 层合并，块间数据必须片上驻留）

### 原则
- **three tier hierarchical dispatch**：根据 BT（块大小）选择不同 kernel：BT=16 直接用 `solve_tril_16x16_kernel`（基础递推）；BT=32 用 `merge_16x16_to_32x32_inverse_kernel`（4 块分治 + 1 次 merge）；BT=64 用 `merge_16x16_to_64x64_inverse_kernel`（16 块分治 + 多层 merge）。每个 kernel 内部保持 16×16 为基本计算单元（NPU 矩阵计算最优粒度）。
- **block recursive inverse formula**：分块求逆递推公式：`X[i,j] = -X[i,i] · Σ_{k=j}^{i-1} A[i,k] · X[k,j]`。与标量递推 `X[i,j] = -A[i,j] - Σ A[i,k]·X[k,j]` 在数学结构上完全一致——前者逐块，后者逐元素。统一性使得代码逻辑在不同层级可复用。
- **divide and conquer merge**：BT=32 将 32×32 矩阵分为 4 个 16×16 块，计算对角块逆后通过 `X₂₁ = -X₂₂ · A₂₁ · X₁₁` 合并。BT=64 分 16 个块，需要 3 层合并（紧邻→隔行→最远），共 7 次 16×16 tl.dot。
- **grid parallelism**：grid = (NT, B×H)，所有 chunk 内的所有 (batch, head) 对完全独立并行。NT = T/BT（定长）或 len(chunk_indices)（变长）。
- **atomic unit 16x16**：所有层级统一以 16×16 为原子计算单元（tl.dot / 逐行递推），匹配 NPU Cube 单元的最优粒度。

### 代码/骨架出处
- `dispatch`:
```python
def solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float):
    B, T, H, BT = A.shape
    assert BT in [16, 32, 64]

    NT = len(chunk_indices) if cu_seqlens else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)

    # 按 BT 选择 kernel
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel

    # grid: (NT 个 chunk, B*H 个 batch-head 组合)
    merge_fn[NT, B * H](A=A, Ai=Ai, ...)
    return Ai
```

**坏实践（反例）**：对大 BT 直接用 O(BT³) 标量递推求逆（或逐元素循环）→ 计算量立方级膨胀，Cube 单元闲置 → ✅ 正解即本条：按 BT 分层 dispatch，统一以 16×16（NPU 最优粒度）为原子单元，分治 merge 降为 O(BT²) 递推 + O(BT) 次 16×16 矩阵乘。

**已知实例**：solve_tril §raw (`experiences/solve_tril/exp_solve_tril_004.yaml`)
- 声称性能收益：通过分治将 O(BT³) 求逆降为 O(BT²) 的递推 + O(BT) 个 16×16 矩阵乘法
- 备注：此架构是算子设计层面的决策，非后期优化。16×16 原子单元的选择由 NPU 矩阵乘法硬件决定。

<!-- okf:related:start -->

# 相关

- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 solve_tril
- 相关主题: [Moe Dispatch Skeleton](../../reference/skeleton-moe_dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](../memory-sram/exp_solve_tril_002.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）](../meta-skills/precompute-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）](../portability/exp_solve_tril_005.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](chunk-recurrence-pingpong-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rope_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: compile time dispatch](exp_radix_convert_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 linear-algebra-operations, solve_tril

<!-- okf:related:end -->
