---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fused add-RMSNorm with weight hoisting"
description: "matmul 场景下，Memory Access Optimization: fused add-RMSNorm with weight hoisting，收益 35%"
tags: [访存效率, reduction-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fused add-RMSNorm with weight hoisting

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，dim≥1024、batch≥32；失效边界：dim 过小无法打满内存总线，融合与 hoisting 收益稀释；权重张量过大无法常驻寄存器/UB 时 hoisting 失效，需改回循环内加载或沿 dim 切块）

### 原则
- **operator fusion**：Fuses the residual addition (hidden_state + residual) and RMSNorm operations into a single kernel. This eliminates the need to write and read back the intermediate addition result from global memory, significantly reducing memory bandwidth pressure.
- **weight hoisting**：The weight tensor is loaded inside the row processing loop. An optimization is to move the weight loading statement outside the `for row_start` loop. Since weights are constant across the batch, this ensures the data is loaded once and kept in registers or cache, minimizing redundant global memory accesses.
- **vectorized reduction**：Utilizes `tl.sum` on the squared values to compute variance efficiently. This maps well to the vector units of the target hardware, allowing for parallel reduction of the `dim` axis.
- **precision control**：Performs the reduction (variance calculation) in `float32` to maintain numerical stability, while keeping the loads and stores in the lower precision `fp16`/`bf16` to optimize memory throughput.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/190-add_gemma_rms_norm_kernel.py
```

**坏实践（反例）**：权重在 row 处理循环内重复加载
```python
for row_start in range(0, batch, BLOCK_M):
    w = tl.load(w_ptr + tl.arange(0, BLOCK_N))  # 每个 row tile 重复读同一份权重
    x = tl.load(x_ptr + row_start * dim + offs)
    ...
```
后果：对 batch 内所有行不变的常量权重被重复 GM 读取 batch/BLOCK_M 次，纯属带宽浪费。✅ 正解：把权重 load 提升到 `for row_start` 循环外，只读一次并驻留寄存器/UB；残差相加与归一化融合为单 kernel，中间和不再落 GM；方差规约用 fp32。

**已知实例**：matmul §raw (`experiences/matmul/exp_gemma_rmsnorm_001.yaml`)
- 声称性能收益：35%
- 备注：该 kernel 实现的是一种缩放权重按 (w + 1.0) 使用的 RMSNorm 变体。适用于 hidden dimension 足够大、能打满内存总线的场景。

<!-- okf:related:start -->

# 相关

- 实践案例: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](../../ops/add-broadcast-alpha.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 实践案例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 matmul
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 同属优化知识，主题相关
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 开发指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 同属优化知识，主题相关
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 同属优化知识，主题相关
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_qkv_rmsnorm_rope_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_rmsnorm_sigmoid_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_layernorm_npu_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorized memory access](../compute-pipeline/exp_reduce_split_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: dynamic tiling strategy](../compute-pipeline/exp_rmsnorm_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_fwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_rmsnorm_kv_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_rmsnorm_rope_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: grid stride loop](../compute-pipeline/exp_sda_bwd_reduce_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [循环不变量按复用组外提（cos/sin 类小表每组一次，实测单点 +100%）](../compute-pipeline/loop-invariant-group-hoisting.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [访存优化：大尺寸 matmul 的 256 分块与 L0C 复用](exp_matmul_001.md) — 主题相关：共享 matmul, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
