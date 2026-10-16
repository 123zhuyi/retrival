---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历
description: 910b2/910b3/91093/91095 上用 tl.parallel(0, 2, bind_sub_block=True) 把 query 头映射到 AICore 内两个向量单元，GQA 每 KV 头加载一次供组内多 query 头复用（带宽降为 1/组大小），block_table 遍历分页 KV，声称收益 45%
tags: [跨芯片适配, attention-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/91093/91095 ｜ triton-ascend 版本 未标注（需支持 bind_sub_block）｜ dtype 混合精度 ｜ shape GQA（q_heads_per_kv_head >1）、分页 KV
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：tl.parallel(bind_sub_block=True) 依赖双向量核结构（910 系列）与后端支持，GPU Triton 无此 API；MHA（组大小=1）时 KV 复用无收益；KV 连续存储（非分页）时 block_table 遍历为额外间接层，可直接算地址；910C/310P 未验证（unverified)）

### 原则
- **vector sub block parallelism**：Utilizes `tl.parallel(0, 2, bind_sub_block=True)` to map the processing of query heads to the two independent vector units within a single AICore. This parallelizes the softmax and reduction operations, effectively doubling the throughput for the vector-heavy parts of the attention mechanism.
- **cube vector pipeline**：Explicitly separates computation into CUBE operations (QK and PV matrix multiplications) and VECTOR operations (softmax, scaling, and accumulation). This decoupling allows the NPU to fully utilize the dedicated Cube unit for high-throughput matrix math while the Vector unit handles the non-linear operations, minimizing pipeline stalls.
- **gqa bandwidth optimization**：Implements Grouped-Query Attention by loading the inputB and inputC tensors once per KV head and reusing them for multiple query heads within the same group. This reduces the memory read bandwidth for KV cache by a factor equal to the group size (q_heads_per_kv_head).
- **paged kv iteration**：Iterates over physical memory blocks using a `block_table` to handle paged KV caches. This allows the kernel to efficiently process long sequences with non-contiguous memory layouts, typical in LLM serving, without requiring data copying or defragmentation.

### 代码/骨架出处
- `TritonPath`:
```python
_paged_gqa_fwd_kernel_high_performance
```

**坏实践（反例）**：GQA 场景每个 query 头独立加载对应 KV → KV cache 读取带宽放大 q_heads_per_kv_head 倍，长序列下访存成为瓶颈 → ✅ 正解即本条：按 KV 头加载一次、组内 query 头复用，配合 `tl.parallel(0, 2, bind_sub_block=True)` 吃满双向量核

**已知实例**：attention §raw (`experiences/attention/exp_gqa_paged_npu_001.yaml`)
- 声称性能收益：45%
- 备注：This kernel is specifically optimized for Ascend NPU architecture (910 series) leveraging the dual-vector core structure. It requires the Triton backend to support `bind_sub_block`.

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 主题相关：共享 attention
- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention
- 实践案例: [LightningIndexer (sparse attention indexer) on Ascend NPU](../../ops/lightning-indexer.md) — 主题相关：共享 attention
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 attention
- 实践案例: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](../../ops/nsa-selected-attention.md) — 主题相关：共享 attention
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Scheduling Optimization: heterogeneous async scheduling](../compute-pipeline/exp_lightning_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse iteration strategy](../compute-pipeline/exp_sparse_attn_topk_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Compute Optimization: sparse data access](../compute-pipeline/exp_sparse_flash_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: kv tile reuse](../memory-sram/exp_attn_bwd_hstu_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: fp8 value casting](../memory-sram/exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: sparse gather optimization](../memory-sram/exp_attn_sparse_bwd_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: simd index select](../memory-sram/exp_flashattn_gqa_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: loop invariant code motion](../memory-sram/exp_lightning_attn_decode_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: paged kv cache](../memory-sram/exp_mla_paged_attn_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](../memory-sram/exp_paged_gqa_001.md) — 主题相关：共享 attention
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](../memory-sram/exp_sdpa_flash_001.md) — 主题相关：共享 attention, attention-operations
- 相关主题: [并行轴选择必须与连续访问对齐，而非与最大 numel 对齐](../memory-sram/parallel-axis-align.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 attention, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 attention, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 attention, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 attention, 跨芯片适配

<!-- okf:related:end -->
