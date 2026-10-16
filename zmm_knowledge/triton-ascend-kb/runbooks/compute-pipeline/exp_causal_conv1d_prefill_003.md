---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: two dimensional grid strategy'
description: 'conv 场景下，Scheduling Optimization: two dimensional grid strategy，收益 2D Grid vs 1D Grid 有 25-39% 性能优势（搜索验证）；整体 Prefill kernel 1.60x 平均加速比'
tags: [多核并行与负载均衡, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: two dimensional grid strategy

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：seq_len 短的 update/decode 场景（seq_len≈1）应选 1D grid + BDT 布局的互补方案；dim < 512、D-tile 数极少时 2D 展开收益有限；寄存器复用依赖 T 维不切分，T 维必须切分的场景此方案不成立）

### 原则
- **two dimensional grid strategy**：采用 2D Grid = `(batch, cdiv(dim, BLOCK_D))` 并行方案。比 1D grid（每条序列一个 program）的优势：(1) 沿 D 维度展开并行，D=4096 时切成 4 个 BLOCK_D=1024 tile，SM 占用率 4 倍提升；(2) 每个 program 的计算密度更合理（1024 维而非 4096 维），避免寄存器溢出；(3) 遗传搜索验证：2D grid 比 1D→2D 方案低 25%，比 1D strided 方案低 39%。
- **btd layout rationale**：Prefill 采用 BTD `[B, T, D]` 布局而非 Update 的 BDT。原因是 Prefill 阶段 seq_len 大（512-4096），BTD 布局下同一 token 的所有 D 维度元素连续存储（stride=1），一次 load 可获取 1024 个连续元素（32 cache lines），硬件预取高效。BDT 布局对长序列的逐 token 处理反而需要跨步访问（行间距 = seq_len），load 效率低。
- **d tile only no t tiling**：仅对 D 维度做 tile 化（BLOCK_D=1024），不对 T 维度做 tile 化——一个 program 处理整条序列的所有 T 个 token。原因是滑动窗口寄存器复用依赖跨 token 的寄存器移位，若 T 维度被切分，tile 边界需要跨 tile 状态传递，寄存器复用的优势会被 tile 边界处理逻辑抵消。
- **grid search validated**：遗传搜索在 2 轮内收敛到 2D Grid 为全局最优：Round 1→Round 2 性能提升 26%（加入 DIM_IS_MULTIPLE + 指针预计算），Round 2→Round 3 提升 <1%（已接近硬件极限）。最终配置为 Route 1（2D Grid）+ Round 2（scalar-reduction），1.60x 平均加速比（最高 1.84x）。

### 代码/骨架出处
- `grid_launch`:
```python
# 2D Grid: batch × D-tiles
grid = (batch, triton.cdiv(dim, BLOCK_D))
kernel[grid](x, weight, conv_states, ...)

@triton.jit
def kernel(..., BLOCK_D: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_d = tl.program_id(1)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
```

**坏实践（反例）**：长序列 prefill 用 1D grid（每条序列一个 program），单 program 顺序处理全部 4096 维 → 单核计算密度过高导致寄存器溢出，且 D 维并行度完全丢失、多核闲置（实测比 2D 方案慢 25-39%）→ ✅ 正解即本条：2D grid `(batch, cdiv(dim, BLOCK_D))` 沿 D 维展开并行。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_prefill_003.yaml`)
- 声称性能收益：2D Grid vs 1D Grid: 25-39% 性能优势（搜索验证）; 整体 Prefill kernel: 1.60x 平均加速比
- 备注：2D Grid + BTD 布局是 Prefill 阶段的架构基础决策，与 Update kernel 的 1D Grid + BDT 布局 + T-tiling 形成互补——两者针对不同的 seq_len 范围（长 vs 短）做了相反的架构选择。选择依据见 tiling.md Layout-Driven Tile Size Selection 节。

<!-- okf:related:start -->

# 相关

- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](../algorithm-numerics/two-pass-reduce-then-apply.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: flattened grid scheduling](../memory-sram/exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: fused state update](../memory-sram/exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: overlapping t tiling](../memory-sram/exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](../memory-sram/exp_gn_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](../memory-sram/exp_rmsnorm_bwd_large_cols_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](../memory-sram/exp_rmsnorm_infer_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations, 多核并行与负载均衡
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](exp_celoss_smooth_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](exp_lightning_attn_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: two pass algorithm](exp_rmsnorm_backward_npu_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: block tiling strategy](exp_sdpa_bwd_kv_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 主题相关：共享 多核并行与负载均衡

<!-- okf:related:end -->
