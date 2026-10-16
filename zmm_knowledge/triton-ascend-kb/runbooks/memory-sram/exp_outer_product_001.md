---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: loop invariant code motion licm"
description: "misc 场景下，Memory Access Optimization: loop invariant code motion licm，收益 45%"
tags: [访存效率, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: loop invariant code motion licm

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp32，M≥1024、N≥1024；失效边界：N 大到 NEED_LOOP_N 为真（rhs 放不进片上缓存）时循环外加载失效，须回退循环内加载；M/N <1024 时 autotune 扫描的 tile（1024–32768）相对过大，收益稀释）

### 原则
- **loop invariant code motion licm**：The kernel employs a conditional branch based on NEED_LOOP_N to optimize memory access. When N is small enough, the rhs vector is loaded once outside the main loop (else block). This drastically reduces global memory read bandwidth for the rhs vector from O(M*N) to O(N), as the loaded data is reused across all M blocks.
- **grid stride loop scheduling**：The kernel utilizes a grid-stride loop (for task_id in range(pid, total_tasks_num, num_jobs)) to process the output matrix. This decouples the total workload from the number of launched cores, allowing the kernel to handle arbitrarily large matrix dimensions (M, N) without exceeding hardware grid limits or requiring kernel re-launches.
- **adaptive tiling strategy**：The @triton.autotune configuration sweeps through a wide range of tile sizes (1024 to 32768) with num_stages=3. This allows the runtime to select the optimal tile size that balances instruction-level parallelism and memory latency hiding, specifically tuning for the L1/L2 cache hierarchy of the target hardware.
- **vectorized boundary handling**：The use of tl.load and tl.store with explicit masks ensures that the kernel handles non-multiple-of-tile dimensions efficiently without padding, maintaining contiguous GM→UB access even at the edges of the matrix.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/140-outer_kernel.py
```

**坏实践（反例）**：N 足够小（rhs 可驻留片上）时仍在每个 M-block 内重复加载 rhs
```python
for task_id in range(pid, total_tasks_num, num_jobs):
    rhs = tl.load(rhs_ptr + n_off, mask=...)   # 每个 M-block 重复加载同一 rhs
    acc = lhs_scalar * rhs
    tl.store(out_ptr + ..., acc)
```
后果：rhs 的全局内存读带宽为 O(M×N)，同一向量被反复从 GM 搬入，成为带宽瓶颈，45% 收益流失。✅ 正解：按 NEED_LOOP_N 条件分支——N 足够小时 rhs 在主循环外加载一次并驻留复用，rhs 读带宽降为 O(N)；阈值依据 N 相对片上缓存/scratchpad 容量设定。

**已知实例**：misc §raw (`experiences/misc/exp_outer_product_001.yaml`)
- 声称性能收益：45%
- 备注：The @libentry decorator suggests this is part of a specialized library entry. The NEED_LOOP_N heuristic is critical; it should be set based on the size of N relative to the cache/scratchpad size to maximize the benefit of hoisting the RHS load.

<!-- okf:related:start -->

# 相关

- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_element_mul_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_index_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](../compute-pipeline/exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](../compute-pipeline/exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_mhc_post_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: micro kernel tiling](../compute-pipeline/exp_tril_solve_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [循环不变量外提（LICM）减少重复地址计算](../compute-pipeline/loop-invariant-code-motion.md) — 同属优化知识，主题相关
- 相关主题: [循环不变量按复用组外提（cos/sin 类小表每组一次，实测单点 +100%）](../compute-pipeline/loop-invariant-group-hoisting.md) — 同属优化知识，主题相关
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
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 misc, 访存效率
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
