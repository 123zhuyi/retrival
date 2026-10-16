---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: kernel fusion"
description: "misc 场景下，Memory Access Optimization: kernel fusion，收益 35%"
tags: [访存效率, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: kernel fusion

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095，fp16/bf16/fp32，n_vocab ≥1024、BLOCK_SIZE ≥128，行主序布局；失效边界：n_vocab 很小或只有单一惩罚项时，融合收益被 kernel launch 开销抵消；非行主序布局时连续访存假设失效，需先调整指针计算）

### 原则
- **kernel fusion**：Fuses frequency penalty, presence penalty, repetition penalty, and temperature scaling into a single kernel launch. This significantly reduces global memory read/write bandwidth by processing all penalties in one pass over the logits tensor.
- **conditional loading**：Implements a check on the `is_present` flag to conditionally load the frequency tensor and execute penalty logic. This avoids unnecessary memory loads and computation for batches that do not contain valid token history (e.g., padding or empty sequences).
- **precision upcast**：Converts inputs to `float32` before performing arithmetic operations. This ensures numerical stability during the division and subtraction operations required by penalty algorithms, preventing precision loss that might occur with `fp16` accumulation.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/226-_fused_penalty_temp_kernel.py
```

**坏实践（反例）**：每项惩罚/缩放各起一个 kernel，整张 logits 反复过 GM
```python
freq_penalty_kernel(logits)      # kernel1：读出 logits，改，写回
presence_penalty_kernel(logits)  # kernel2：再读再写
temperature_kernel(logits)       # kernel3：又一轮完整读写
```
后果：logits 在 GM 往返次数随惩罚项数线性放大，带宽成为瓶颈。✅ 正解：单 kernel 单 pass 内联全部惩罚与温度缩放；配合 `is_present` 条件加载，跳过无有效历史（padding/空序列）的 batch 的内存读取与计算。

**已知实例**：misc §raw (`experiences/misc/exp_logits_penalty_fusion_001.yaml`)
- 声称性能收益：35%
- 备注：kernel 假设 logits 与频率张量均为行主序布局。BLOCK_SIZE 应按目标硬件的访存粒度（如 128B 或 256B）调优以获得最佳带宽。

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_element_mul_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_index_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](../compute-pipeline/exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](../compute-pipeline/exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](../compute-pipeline/exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_mhc_post_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: micro kernel tiling](../compute-pipeline/exp_tril_solve_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](../portability/exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](../portability/exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](../portability/exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: indirect load scalarization](exp_grouped_attn_stage1_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_hstu_attn_fwd_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 matrix-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](exp_paged_attn_flash_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](exp_paged_gqa_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: paged kv indirection](exp_paged_prefill_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](exp_rope_forward_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: sparse iteration logic](exp_sparse_attn_backward_dq_001.md) — 主题相关：共享 matrix-operations, 访存效率
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, 访存效率

<!-- okf:related:end -->
