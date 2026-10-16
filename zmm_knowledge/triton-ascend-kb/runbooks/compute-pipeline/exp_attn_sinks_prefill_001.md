---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: attention sinks mechanism'
description: 'attention 场景下，Compute Optimization: attention sinks mechanism，收益 35%'
tags: [AICore 单元利用/Scalar 削减, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: attention sinks mechanism

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：强绑定 Br==16、PAGE_SIZE==16 的 tile 假设，页大小或查询块变化需重新调优；依赖 block_tables 页表，非 paged 内存布局不适用；无 sink token、无滑窗、连续缓存的简单场景应选更轻量的 kernel）

### 原则
- **attention sinks mechanism**：The kernel implements the Attention Sinks technique by initializing the `history_max` variable with a pre-loaded `sink` value. During the online softmax loop, the accumulator `acc` and normalization factor `l` are updated incrementally. Finally, the sink's contribution is explicitly added back to `l` and used to rescale `acc`. This ensures the model retains attention to the initial token during streaming generation, mitigating attention drift.
- **online softmax algorithm**：Instead of materializing the full $inputA \times inputB^T$ attention matrix, the kernel employs an online softmax approach. It iterates over blocks of the KV cache, computing partial dot products (`qk`) and updating the running maximum (`history_max`), exponential sum (`l`), and weighted value accumulator (`acc`) using rescaling factors (`re_scale`). This significantly reduces global memory access and register pressure, which is critical for long-context prefill.
- **sliding window pruning**：The kernel calculates `start_page_num` and `start_kv_len_block` based on the `sliding_window_size` parameter. By masking out KV pages that fall outside the sliding window (using `mask_page`), it avoids loading and computing attention for irrelevant history tokens, thereby optimizing memory bandwidth and compute utilization.
- **paged kv cache access**：It utilizes `block_tables` to map logical KV pages to physical memory blocks. This allows the kernel to handle non-contiguous memory layouts efficiently, supporting advanced memory management techniques like paged attention used in vLLM, which is essential for serving long-context models with limited VRAM.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/174-attention_sinks_prefill_kernel.py
```

**坏实践（反例）**：在主归约循环里用分支单独处理 sink token，或归约结束后再单独跑一遍 kernel 补 sink 贡献 → 循环内分支造成执行分歧，额外 pass 又多读一遍数据 → ✅ 正解即本条：用 sink 值初始化 history_max，把 sink 贡献合并进最终归一化步（`e_sum += tl.exp(sink - e_max)`）。

**已知实例**：attention §raw (`experiences/attention/exp_attn_sinks_prefill_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is optimized for the prefill stage where query sequences are processed in blocks of 16 (Br). It requires `cum_seq_lens` and `block_tables` to manage variable-length sequences and paged memory.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: vectorized offset generation](../memory-sram/exp_attn_causal_mask_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: block pointer abstraction](../memory-sram/exp_attn_ptr_setup_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](../memory-sram/exp_chunk_gated_delta_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: tiling strategy](../memory-sram/exp_flash_attn_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: indirect load scalarization](../memory-sram/exp_grouped_attn_stage1_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_hstu_attn_fwd_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: block ptr utilization](../memory-sram/exp_paged_attn_decode_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: Paged Memory Access Strategy](../memory-sram/exp_paged_attn_flash_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: paged kv indirection](../memory-sram/exp_paged_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](../memory-sram/exp_rope_forward_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: block ptr abstraction](../memory-sram/exp_sdpa_infer_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](../portability/exp_attn_bwd_q_down_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](../portability/exp_attn_sda_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](../portability/exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic autotuning](exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](exp_block_matmul_inv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: pointer base precompute](exp_causal_conv1d_prefill_004.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: conv bias silu triple fusion](exp_causal_conv1d_prefill_006.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: w loop unrolling with threshold](exp_causal_conv1d_update_002.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block ptr tiling](exp_chunk_fwd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](exp_chunk_kkt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](exp_cube1_qk_gemm_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_flashattn_bwd_kv_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: split k parallelism](exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: Operator Fusion](exp_fused_delta_rule_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: deep kernel fusion](exp_fused_delta_rule_npu_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_entropy_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gemm_scale_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](exp_kda_bwd_intra_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: 对角块标量循环改 tl.dot](exp_kda_bwd_scalar_loop_to_dot_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [标量消除: i64/i32 标量降级定向消除](exp_kda_bwd_wy_scalar_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [计算优化: phase split 与 barrier 外提](exp_kda_bwd_wy_structure_opt_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: npu simd vectorization](exp_layernorm_npu_simd_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: block tiling strategy](exp_matmul_bias_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: small tile strategy](exp_matmul_triton_base_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](exp_matmul_vec_parallel_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](exp_partial_matmul_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: fused rope computation](exp_rope_matmul_fused_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: vectorized non interleaved path](exp_rope_siso_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: autotuning block size](exp_sdpa_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: online softmax](exp_sdpa_infer_ascend_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention, matrix-operations
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](exp_solve_tril_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](exp_solve_tril_003.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: sparse iteration strategy](exp_sparse_attn_topk_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: sparse data access](exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, attention
- 相关主题: [Compute Optimization: kernel fusion](exp_ssm_gated_delta_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, matrix-operations

<!-- okf:related:end -->
