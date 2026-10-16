---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: dynamic block sizing with fused backward+add"
description: "normalization 场景下，Memory Access Optimization: dynamic block sizing with fused backward+add，收益 25%"
tags: [访存效率, reduction-operations, normalization]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: dynamic block sizing with fused backward+add

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，fp16/bf16/fp32，反向梯度与残差梯度相加融合的归一化场景；失效边界：寄存器累加 dW 要求整行 fp32 累加器可驻留：n_cols × 4B ≤ 单核寄存器/UB 可用预算，n_cols 过大导致寄存器溢出时应改用 pid 分区 buffer 方案；n_rows 很小、grid-stride 退化为单轮时负载均衡收益消失）

### 原则
- **dynamic block sizing**：The heuristic adjusts BLOCK_SIZE_M based on n_cols to maintain a constant tile area (~4096 elements), optimizing L1/L0 cache utilization and register pressure regardless of the hidden dimension size.
- **operator fusion**：Fuses the normalization backward pass with the addition of the residual gradient (dS_out), reducing global memory read/write operations by avoiding an intermediate kernel launch.
- **register accumulation**：Accumulates the weight gradient (dW) in a register-level float32 buffer (dW_acc) across all processed rows, writing to global memory only once per program ID to minimize memory bandwidth.
- **grid stride scheduling**：Implements a grid-stride loop for row processing, ensuring efficient load balancing across NPU cores even when the number of rows is not perfectly divisible by the block size.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/249-_fused_add_rmsnorm_bwd_kernel.py
```

**坏实践（反例）**：残差梯度相加与反向计算拆成两个 kernel，中间结果写回 GM 再读回
```python
kernel_a: dS_out = dY + dR; store dS_out to GM   # 中间张量落 GM
kernel_b: load dS_out from GM; compute dX, dW    # 再次读回
```
后果：中间张量多一次 GM 写 + 一次 GM 读，并多付一次 kernel launch 开销。✅ 正解：单 kernel 内融合相加与反向，中间结果留在寄存器/SRAM；dW 用 fp32 寄存器累加器跨行累加，每个 pid 仅写 GM 一次（收益 25%）。

**已知实例**：normalization §raw (`experiences/normalization/exp_rmsnorm_bwd_fused_001.yaml`)
- 声称性能收益：25%
- 备注：支持多种模型特定的精度转换（cast）模式以满足混合精度需求。The heuristic ensures robustness across varying sequence lengths and hidden sizes.

<!-- okf:related:start -->

# 相关

- 实践案例: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](../../ops/add-broadcast-alpha.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 开发指南: [Embedding / 索引拷贝算子优化](../../reference/op-class-embedding.md) — 同属优化知识，主题相关
- 开发指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../compute-pipeline/load-balance-grid-sizing.md) — 同属优化知识，主题相关
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic workload tiling](exp_layernorm_bwd_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: persistent kernel strategy](exp_layernorm_fwd_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: grid stride scheduling](exp_rmsnorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: backward column tiling with conflict-free dW accumulation](exp_rmsnorm_backward_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: two-pass reduction for large columns](exp_rmsnorm_bwd_large_cols_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D vector loading for small hidden dims](exp_rmsnorm_fused_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: 2D block tiling with partial-sum reduction](exp_rmsnorm_grad_dw_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: no-tiling single-load for small rows](exp_rmsnorm_notiling_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column blocking for UB management](exp_rmsnorm_npu_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: column tiling for wide rows](exp_rmsnorm_tiled_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: multibuffer software pipelining](exp_sdpa_bwd_reduce_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: precision control](exp_skip_layernorm_001.md) — 主题相关：共享 normalization, reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: block size tuning](exp_sum_scale_001.md) — 主题相关：共享 reduction-operations, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 reduction-operations, 访存效率

<!-- okf:related:end -->
