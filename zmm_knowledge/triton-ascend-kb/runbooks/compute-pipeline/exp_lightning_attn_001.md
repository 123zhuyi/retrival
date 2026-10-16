---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.3
title: 'Scheduling Optimization: heterogeneous async scheduling'
description: 'attention 场景下，Scheduling Optimization: heterogeneous async scheduling，收益 45%'
tags: [多核并行与负载均衡, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Scheduling Optimization: heterogeneous async scheduling

【优化维度】：多核并行与负载均衡
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：依赖 `dl` dialect 与 cross_flag 硬件同步原语，无该扩展不可用；decay 因子 workspace 占用片上缓冲，块过大时溢出；短序列下线性复杂度的优势不抵 kernel 结构开销，二次方方案可能更优）

### 原则
- **heterogeneous async scheduling**：The kernel utilizes `dl.async_task` to explicitly separate vector operations (decay factor calculation) into the Vector unit and matrix multiplication operations into the Cube unit. This allows the hardware to execute these pipelines in parallel, hiding the latency of scalar/vector computations behind the heavy matrix math.
- **hardware specific synchronization**：Implements `dl.set_cross_flag(dl.SyncFlag.V2C, 0)` to manage synchronization between the Vector unit (producer of decay factors) and the Cube unit (consumer). This ensures data readiness without stalling the pipeline, optimizing the data flow between different NPU processing elements.
- **workspace reuse strategy**：Pre-calculates complex exponential decay factors (`q_decay`, `k_trans_decay`, `block_decay`, `diag_decay`) once per block and stores them in a designated workspace. This avoids redundant transcendental function calls (`tl.exp`) inside the main computation loop, significantly reducing instruction count.
- **block wise state accumulation**：Optimizes the linear attention recurrence by maintaining a running state `kv` in registers/local memory. The update rule `kv = block_decay * kv + dot(k_trans * k_trans_decay, v)` is fused within the loop, minimizing global memory access for the KV cache and maximizing data reuse.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/068-_fwd_loop_kernel.py
```

**坏实践（反例）**：主循环内反复调用 `tl.exp` 重算指数衰减因子 → transcendental 指令主导循环开销 → ✅ 正解即本条：decay 因子每块预计算一次存 workspace 复用 + Vector（decay 计算）与 Cube（矩阵乘）用 set/wait_cross_flag 流水重叠。

**已知实例**：attention §raw (`experiences/attention/exp_lightning_attn_001.yaml`)
- 声称性能收益：45%
- 备注：This kernel is specifically designed for the Ascend NPU architecture using the `dl` dialect. It is highly effective for long-context scenarios (large n) where the linear complexity of linear-attention mechanisms outperforms standard quadratic attention.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: kv tile reuse](../memory-sram/exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: fp8 value casting](../memory-sram/exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: sparse gather optimization](../memory-sram/exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: simd index select](../memory-sram/exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: loop invariant code motion](../memory-sram/exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: paged kv cache](../memory-sram/exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](../portability/exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](async-task-cube-vector.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
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
- 相关主题: [Scheduling Optimization: grid stride loop](exp_flash_attn_gridstride_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [流程优化: 标准 score-归一化-加权三步流程](exp_fused_attention_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_attn_preprocess_001.md) — 主题相关：共享 attention, 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: async pipeline scheduling](exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rms_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Scheduling Optimization: operator fusion](exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 多核并行与负载均衡
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention, attention-operations

<!-- okf:related:end -->
