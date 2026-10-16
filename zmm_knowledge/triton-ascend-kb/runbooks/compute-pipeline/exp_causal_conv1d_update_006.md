---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.4
title: 'Cross-kernel Optimization Transfer: token loop vs w loop decision'
description: 'conv 场景下，Cross-kernel Optimization Transfer: token loop vs w loop decision，收益 Bias 融合消除 1 次 kernel launch（约 10-20% 延迟）、指针预计算减少 5-10% 标量指令'
tags: [融合模式, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Cross-kernel Optimization Transfer: token loop vs w loop decision

【优化维度】：融合模式
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：迁移仅在源/目标 kernel 的循环维度与布局前提兼容时成立；W≠4、或 T_CHK_SIZE 与 seq_len 接近时 token-loop 特化不适用；W 硬编码与 2D Grid 这类与架构决策耦合的优化明确不可迁移）

### 原则
- **token loop vs w loop decision**：Prefill kernel 在 Token 维度上迭代（seq_len 次，每次处理 BLOCK_D=1024 维），Update kernel 在 W 维度上迭代（width 次，每次处理 D_CHK_SIZE=16 维）。对于 W=4 的典型场景，Prefill 的 Token 循环方案在算术效率上有根本优势：(1) 零数据编排开销（无 extract_slice/transpose，Update 有 11 次）；(2) 更好的 FMA 映射（单语句 4 路归约 vs 4 次独立累积）；(3) BTD 布局下一次 load 1024 个连续元素（32 cache lines）vs BDT 布局下 16 行分散加载。但这不意味着 Prefill 方案在所有场合都更好——seq_len=1 时 BDT+W 循环更适合单 token decode 的内存访问模式。
- **bias fusion transfer**：从 Prefill kernel 迁移 HAS_BIAS 支持到 Update kernel。Prefill 在卷积累加后直接 `acc = acc + bias`（bias 预加载为寄存器），Update kernel 中 HAS_BIAS 已声明为 `tl.constexpr` 但对应实现在源码中尚未落地（留有待办标记）。实现仅需在 W 循环或展开表达式的结果 `out_block` 上加 `bias[None, :]`。消除额外的 bias add kernel launch。
- **pointer base precompute**：从 Prefill kernel 的 `x_ptrs = x_ptr + start*stride + d_offs*stride_dim` 模式迁移到 Update kernel 的 W 循环内部。在 `for owi in range(width)` 循环中，x_b 切片和 w 切片的基址可预计算，减少每次 extract_slice 中的偏移运算。适用于未展开 W 循环的原始路径。
- **single token specialization**：当 seq_len=1（单 token decode）时，可将矩阵式卷积退化为 Prefill 风格的滑动窗口寄存器复用：预先加载 col0/col1/col2/col3 和 w0/w1/w2/w3 到寄存器，直接计算 `out = col0*w0 + col1*w1 + col2*w2 + col3*w3`，然后移位 `col0=col1; col1=col2; col2=col3; col3=new_token`。避免了为 1 个 token 加载整个 [D_CHK, 1+W-1] buffer 的浪费。优先级低，因为 T_CHK_SIZE=256 >> seq_len=1 时改动较大。
- **not applicable transfers**：以下 Prefill 优化不建议迁移到 Update：(1) W 硬编码为 4 — Update 需要通用 W 支持；(2) 2D Grid 并行 — Update 场景 seq_len 短，T-tile 切分是必要的，1D strided grid 配合 3D 任务分解更适合。

### 代码/骨架出处
- `bias_fusion`:
```python
# Update kernel 中实现 Bias 融合（从 Prefill 迁移）
if HAS_BIAS:
    bias = tl.load(bias_ptr + d_offs, mask=mask_d, other=0.0)
    out_block = out_block + bias[None, :]
```
- `single_token_specialization`:
```python
# seq_len=1 时的滑动窗口寄存器复用（概念迁移）
col0, col1, col2, col3 = load_state_registers()
w0, w1, w2, w3 = load_weight_registers()
new_token = tl.load(x_ptr)
out = col0*w0 + col1*w1 + col2*w2 + col3*w3
col0, col1, col2 = col1, col2, col3
col3 = new_token
store_state(col0, col1, col2)
```

**坏实践（反例）**：不加甄别地把 Prefill kernel 的全部优化搬到 Update kernel（如 2D Grid、W 硬编码为 4）→ 短序列下 T-tile 并行度丢失、通用 W 支持被破坏，收益为负 → ✅ 正解即本条：按「循环维度 × 布局」前提逐项评估，只迁 bias 融合、指针预计算、mask 消除这类与架构决策解耦的优化。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_update_006.yaml`)
- 声称性能收益：Bias fusion: 消除 1 次 kernel launch（延迟降低约 10-20%）；Pointer precompute: 减少 5-10% 标量指令；Token-loop for seq_len=1: 预计 15-25% 提升（低优先级）
- 备注：此经验是关于 Prefill 和 Update 两个 kernel 之间的优化知识迁移。核心启示：循环维度选择（Token vs W）是根本性的架构决策，应根据 seq_len 和 D 的相对大小选择。Bias 融合和 DIM_IS_MULTIPLE mask 消除是最容易迁移且收益明确的优化。

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](../../ops/moe-token-permute-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](../../ops/moe-token-permute-with-ep-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithRoutingMap (routingMap-driven token permute) on Ascend NPU](../../ops/moe-token-permute-with-routing-map.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](../../ops/moe-token-unpermute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](../../ops/moe-token-unpermute.md) — 同属优化知识，主题相关
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: flattened grid scheduling](../memory-sram/exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: sliding window register reuse](../memory-sram/exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: fused state update](../memory-sram/exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](../memory-sram/exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: overlapping t tiling](../memory-sram/exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv
- 相关主题: [跨 run benchmark 噪声鉴别纪律（framework 侧 3-7x 波动，speedup 跨 run 对比无效）](../meta-skills/cross-run-benchmark-noise.md) — 同属优化知识，主题相关
- 相关主题: [形状驱动算子决策树](../meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations

<!-- okf:related:end -->
