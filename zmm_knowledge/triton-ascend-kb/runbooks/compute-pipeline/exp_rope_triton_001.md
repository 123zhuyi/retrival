---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: vectorized memory access'
description: 'rope 场景下，Compute Optimization: vectorized memory access，收益 25%'
tags: [AICore 单元利用/Scalar 削减, activation-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: vectorized memory access

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：TOKEN_BLOCK_SIZE 需匹配 L1 容量（备注自述，典型 128/256）；cos/sin 复用的前提是各头共享同一位置编码表；interleaved/non-interleaved 由编译期标志决定，布局混用需两个编译版本）

### 原则
- **vectorized memory access**：The kernel utilizes `tl.load` with vectorized offsets (`feat_offset`) to load contiguous chunks of the input tensor `X` into SRAM/Registers. This maximizes memory bandwidth utilization by coalescing memory requests, which is critical for high-throughput hardware like the 910 series.
- **fused computation pipeline**：The rotation arithmetic (multiplication by sin/cos and subsequent addition/subtraction) is fused directly into the kernel execution flow. This eliminates the need to write intermediate results back to global memory (HBM), significantly reducing memory latency and increasing arithmetic intensity.
- **masked execution**：The implementation employs `seq_mask` during both load and store operations. This allows for efficient handling of variable sequence lengths and padding without expensive dynamic branching or separate kernel launches for different sequence lengths.
- **embedding reuse**：The COS and SIN embeddings are loaded once per sequence block (outside the head loop) and reused for all NUM_Q_HEADS and NUM_K_HEADS. This drastically reduces global memory bandwidth consumption, which is the primary bottleneck for RoPE operations in models with many attention heads.
- **grid stride scheduling**：The kernel utilizes a grid stride loop (range(pid, NUM_BLOCKS, NUM_CORES)) to distribute work across available cores. This ensures efficient load balancing and avoids idle cores when the number of sequence blocks is not a multiple of the core count.
- **vectorized block processing**：The kernel processes the sequence dimension in chunks of BLOCK size, enabling efficient memory coalescing for the inputA and inputB tensor loads/stores and maximizing the utilization of the memory bus.
- **grid stride scheduling**：Implements a grid-stride loop (range(pid, NUM_BLOCKS, NUM_CORES)) to distribute sequence blocks across available cores. This ensures balanced workload distribution and high hardware utilization even when the number of sequence blocks is not a multiple of the core count.
- **cos sin reuse**：Loads the COS and SIN tensors once per sequence block, outside the inner loop iterating over NUM_HEADS. Since the rotary embedding values are identical for all heads at a given position, this strategy significantly reduces global memory read bandwidth by a factor of NUM_HEADS.
- **vectorized memory access**：Utilizes a block size (BLOCK) to load and store data along the sequence dimension. By calculating offsets with pos_offset[:, None], the kernel ensures contiguous memory access patterns, maximizing DRAM bandwidth utilization and cache efficiency.
- **grid stride loop**：Implements a grid-stride loop (`for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES)`) to map work to hardware cores. This allows the kernel to process sequence lengths significantly larger than the grid size, ensuring all cores are utilized efficiently and reducing the overhead of kernel launches for long sequences.
- **dimensional chunking**：Strategically partitions the feature dimension `DIM` into four segments (`ll`, `lh`, `hl`, `hh`) based on the `CS_DIM` parameter. This enables efficient vectorized memory loads and stores for both the rotated sub-dimensions and the pass-through sub-dimensions, optimizing memory bandwidth usage.
- **embedding table reuse**：Optimizes global memory access by loading the Cosine (`COS`) and Sine (`SIN`) tables only once per block and reusing the loaded values for both the lower and upper halves of the feature dimension (`cos_h = cos_l`, `sin_h = sin_l`), effectively halving the bandwidth required for embedding lookups.
- **masked memory access**：Utilizes dynamic masking (`pos_mask`, `seq_mask`) to handle sequence lengths that are not multiples of the block size `BLOCK`. This ensures safe memory access and correctness for variable-length sequences without the overhead of explicit padding.
- **vectorized instruction mapping**：The rotation calculation (x1*cos - x2*sin, x2*cos + x1*sin) is mapped to the hardware's vector units. By processing multiple data points simultaneously using wide vector registers (e.g., 128-bit or 256-bit), the instruction throughput is significantly increased compared to scalar operations.
- **broadcast efficiency**：The `cos` and `sin` tensors are expanded to match the batch and sequence dimensions. The optimization strategy involves loading these small tensors into registers or high-speed cache (SRAM) once per compute block and reusing them across the sequence length, drastically reducing global memory bandwidth consumption.
- **data type alignment**：The kernel is optimized for low-precision arithmetic (fp16/bf16) which is standard for LLM inference/training. This ensures full utilization of the NPU's compute capabilities, as the complex multiplication pattern fits well within the supported data type pipelines.
- **mixed precision computation**：The kernel explicitly casts the cos and sin lookup tables to float32 (.to(tl.float32)) before performing the rotation arithmetic. This ensures high numerical stability during the complex multiplication (x*cos - y*sin), which is critical for maintaining model accuracy in LLMs, even when the input state tensors are stored in lower precision formats like fp16 or bf16.
- **vectorized pair loading**：By calculating offsets for dim_range_x and dim_range_y simultaneously, the kernel loads the paired state values (real and imaginary components) required for rotation in parallel. This vectorized approach maximizes memory bandwidth utilization and reduces the number of load instructions compared to processing dimensions sequentially.
- **adaptive layout handling**：The kernel efficiently supports both interleaved (x, y, x, y) and non-interleaved (x...x, y...yaved) memory layouts via the rotary_interleaved flag. It adjusts the sin_sim_offset calculation logic without requiring expensive data transposition kernels prior to the embedding operation.
- **boundary safe masking**：The use of dynamic masks for token_range and head_range ensures that the kernel correctly processes variable sequence lengths and batch sizes without out-of-bounds memory access, eliminating the need for input padding to fixed dimensions.
- **slice based memory access**：Utilizes tl.extract_slice and tl.insert_slice to efficiently handle the tensor splitting and merging required for RoPE. This abstracts complex pointer arithmetic, allowing the compiler to optimize for memory coalescing and reducing the risk of bank conflicts when accessing the x1 and x2 segments.
- **compile time specialization**：Leverages tl.constexpr for head_num, half_rope_dim, and TOKEN_BLOCK_SIZE. This enables the compiler to generate specialized code for specific tensor shapes, facilitating aggressive loop unrolling and optimal register allocation to minimize instruction overhead.
- **compile time branching**：The inverse parameter is defined as a constexpr, ensuring that the conditional logic for forward versus inverse rotation is resolved at compile time. This completely eliminates runtime branch divergence within the processing blocks, maintaining high pipeline utilization.
- **vectorized rotation math**：Implements the rotation logic (x1*cos +/- x2*sin) using direct tensor operations. This maps efficiently to the hardware's SIMD/vector units, maximizing throughput for the element-wise floating-point arithmetic.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/227-_compute_rope.py
```

**坏实践（反例）**：cos/sin 表在头维循环内每次重新 load，旋转中间结果写回 HBM → 位置编码表多读 NUM_HEADS 倍 + 中间结果往返 → ✅ 正解即本条：cos/sin 按块一次加载全头复用 + 旋转算术融合进 load/store 流水 + 低精度存储 fp32 计算。

**已知实例**：rope §raw (`experiences/rope/exp_rope_triton_001.yaml`)
- 声称性能收益：25%
- 备注：This implementation is highly efficient for LLM inference. Ensure TOKEN_BLOCK_SIZE is tuned to match the L1 cache size of the specific hardware (e.g., 128 or 256) to minimize global memory accesses.

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_add_layernorm_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fused_norm_rope_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_geglu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: cos sin reuse](../memory-sram/exp_rope_npu_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: two pass reduction strategy](../memory-sram/exp_softmax_backward_non_inner_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Memory Access Optimization: grid stride loop](../memory-sram/exp_swiglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: blocking strategy](exp_attn_qk_cube_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_fused_gating_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride loop](exp_geglu_flat_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotune block m](exp_gelu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_gelu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: two pass algorithm](exp_log_softmax_backward_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized memory access](exp_partial_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Compute Optimization: fused mrope logic](exp_qwen2vl_mrope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Compute Optimization: cos sin reuse](exp_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope
- 相关主题: [Scheduling Optimization: grid stride loop](exp_rope_backward_001.md) — 主题相关：共享 activation-operations, rope
- 相关主题: [Compute Optimization: kernel fusion](exp_rope_mqa_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: grid stride scheduling](exp_sigmoid_grid_stride_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: autotuning strategy](exp_silu_bwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Scheduling Optimization: grid stride loop](exp_silu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: vectorized accumulation strategy](exp_softmax_inner_002.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: online softmax algorithm](exp_softmax_non_inner_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: activation recomputation](exp_swiglu_bwd_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: software pipelining](exp_swiglu_bwd_flat_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Scheduling Optimization: mixed precision computation](exp_swiglu_fwd_001.md) — 主题相关：共享 activation-operations
- 相关主题: [Compute Optimization: interleaved vectorization](exp_swiglu_oai_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: operator fusion](exp_swiglu_quant_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减
- 相关主题: [Compute Optimization: vectorized chunked processing](exp_triton_rope_001.md) — 主题相关：共享 activation-operations, aicore 单元利用/scalar 削减, rope

<!-- okf:related:end -->
