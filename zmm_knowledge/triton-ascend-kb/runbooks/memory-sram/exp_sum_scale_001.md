---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: block size tuning"
description: "reduction 场景下，Memory Access Optimization: block size tuning，收益 30%"
tags: [访存效率, reduction-operations, reduction]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: block size tuning

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16/fp32，N ≥1024；失效边界：N >10000 时单 program（grid=1）处理全数组成为瓶颈，需重构为多核并行归约；BLOCK_N 增大受 UB 容量约束（BLOCK_N × dtype 字节数 × 数据流条数 ≤ UB 预算，启用权重双数据流时预算减半））

### 原则
- **block size tuning**：Increase blockN from 128 to 1024 or 2048. This aligns better with the memory burst length and cache line size of the Ascend 910B series, significantly reducing the number of memory transactions and loop control overhead.
- **vectorized access**：Ensure that the tl.load operations for both input and weight pointers utilize the maximum vector width (e.g., 128 bytes) supported by the hardware. This maximizes memory bandwidth utilization, which is critical as the kernel loads two data streams simultaneously when scalebyw is enabled.
- **loop unrolling**：Apply loop unrolling to the main accumulation loop to minimize branch prediction penalties and instruction dispatch overhead, allowing the hardware pipeline to process data more efficiently.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/114-sum_and_scale.py
```

**坏实践（反例）**：BLOCK_N 取 128 的小块累加循环，且未用满向量宽度
```python
BLOCK_N = 128
for off in range(0, N, BLOCK_N):              # N=8192 时需 64 趟迭代
    x = tl.load(x_ptr + off + tl.arange(0, BLOCK_N))
    w = tl.load(w_ptr + off + tl.arange(0, BLOCK_N))   # 双数据流均为小块
    acc += tl.sum(x * w)
```
后果：访存事务数与循环控制开销随迭代次数成倍放大，GM→UB 传输无法对齐内存突发长度，带宽利用率低。✅ 正解：BLOCK_N 提升到 1024/2048，对齐 Ascend 910B 系列的内存突发长度与缓存行；两路 `tl.load` 用满硬件向量宽度（如 128B）；主累加循环做展开，压低分支与指令派发开销。

**已知实例**：reduction §raw (`experiences/reduction/exp_sum_scale_001.yaml`)
- 声称性能收益：30%
- 备注：The current implementation processes the entire array within a single program block (grid size 1). For large-scale arrays (N > 10000), it is highly recommended to refactor the kernel into a multi-block parallel reduction pattern to fully utilize the NPU cores.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: vectorization alignment](../compute-pipeline/exp_backward_sum_001.md) — 主题相关：共享 reduction, reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_smooth_001.md) — 主题相关：共享 reduction-operations
- 相关主题: [Compute Optimization: autotuning block size](../compute-pipeline/exp_sdpa_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
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
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
