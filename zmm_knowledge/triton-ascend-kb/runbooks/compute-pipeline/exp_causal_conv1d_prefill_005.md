---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: pad slot early exit'
description: 'conv 场景下，Scheduling Optimization: pad slot early exit，收益 在有 padding 序列的 batch 中节省无效计算（如 25% padding→~25% 资源节省）；最小开销仅 1 load + 1 cmp'
tags: [多核并行与负载均衡, convolution-operations, conv]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: pad slot early exit

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：batch 中无 padding 序列（满载）时早退检查是纯开销（虽仅 1 load + 1 cmp）；kernel 计算量本身极小（几十条指令级）时节省可忽略；batch→slot 无需间接索引映射的场景不适用）

### 原则
- **pad slot early exit**：在 kernel 入口处检查 `cache_idx == pad_slot_id`（-1），匹配时立即 return，不做任何计算。释放对应的 compute unit 给其他有用的 program，提升整体硬件利用率。
- **continuous batching support**：通过 cache_indices 间接映射 batch_id → cache_slot，支持：(1) beam search 场景多条序列共享同一 cache slot；(2) 动态 batch 大小（pad_slot_id=-1 标记无效序列）；(3) 无需为 padding 序列分配额外物理资源。
- **zero cost skip**：`tl.load(cache_indices_ptr + pid_batch)` + `if ... return` 的开销仅为 1 次 load + 1 次整数比较，远小于执行完整卷积循环的成本（~seq_len 次 load/store/FMA）。在 batch 中有 ≥10% 填充序列时收益显著。
- **applicability general**：适用于所有使用间接索引映射 batch 的 kernel（如 attention with KV-cache page table、embedding lookup with dynamic indices、MoE routing），只要能用单次 load + compare 判断是否需要跳过计算。

### 代码/骨架出处
- `early_exit`:
```python
cache_idx = tl.load(cache_indices_ptr + pid_batch)
if cache_idx == pad_slot_id:   # -1
    return                      # 立即释放 compute unit
```

**坏实践（反例）**：对 padding 序列也执行完整计算循环，仅靠输出 mask 丢弃结果 → 无效 program 占满 compute unit，batch 有 25% padding 就浪费约 25% 算力 → ✅ 正解即本条：kernel 入口 load 间接索引、判到 pad_slot_id 立即 return。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_prefill_005.yaml`)
- 声称性能收益：在有 padding 序列的 batch 中节省无效计算比例（如 25% padding→~25% 资源节省）；最小开销仅 1 load + 1 cmp
- 备注：pad_slot 机制不仅节省无效计算，更重要的是让 SM/SIMT 单元可被其他有用工作抢占——在 Ascend 的 grid-stride 调度下，提前退出的 program 对应的 tile 可被重新分配给其他 tasks。

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: flattened grid scheduling](../memory-sram/exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: fused state update](../memory-sram/exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: overlapping t tiling](../memory-sram/exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv
- 相关主题: [非 2 的幂的语义维度必须 pad 到 next_pow2 + 极值填充 + 截断](../memory-sram/pad-next-pow2-extremum.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: 2D Block Tiling](exp_embedding_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](exp_entropy_backward_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](exp_entropy_backward_d_weight_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_entropy_backward_fused_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [pad slot 单次检查提前退出，跳过无效填充序列](pad-slot-early-exit.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
