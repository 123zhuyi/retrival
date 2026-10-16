---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加
description: 910b2/910b3 上 MatMul 用 range(pid, NUM_BLOCKS, NUM_CORES) 静态调度 + grouped_launch_diagonal 对角遍历提升 L2 局部性；dl.compile_hint(..., "dot_pad_only_k") 只 pad K 维避免整 tile padding 开销；tl.dot 第三参传累加器做寄存器原位累加，声称收益 35%
tags: [跨芯片适配, matrix-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（依赖 dl 方言 compile_hint 与外部 grouped_launch_diagonal 函数）｜ dtype 全部 ｜ shape 大矩阵（访存受限）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：dl.compile_hint("dot_pad_only_k") 是 Ascend dl 方言专用，GPU Triton 无此 hint；grouped_launch_diagonal 为外部依赖，未定义时需自行实现 block→坐标映射；小矩阵（计算受限而非访存受限）下对角遍历的 L2 收益不明显；NUM_BLOCKS < NUM_CORES 时静态步长调度部分核空转；910C/310P 未验证（unverified)）

### 原则
- **scheduling strategy**：The kernel implements a static scheduling strategy using a stride loop `range(pid, NUM_BLOCKS, NUM_CORES)` combined with a `grouped_launch_diagonal` function. This diagonal traversal pattern optimizes cache locality by ensuring that blocks processed by different cores are adjacent in memory, thereby reducing memory access contention and improving L2 cache hit rates.
- **compile hint optimization**：The code utilizes `dl.compile_hint` with the argument `"dot_pad_only_k"`. This is a critical hardware-specific optimization that instructs the compiler to pad the input tensors exclusively along the inputB-dimension to satisfy the alignment requirements of the hardware's matrix multiplication unit (e.g., the Ascend Cube Unit). This avoids the memory overhead and computational waste associated with padding the entire M and N dimensions.
- **boundary handling**：Efficient handling of matrix dimensions that are not multiples of the block size using `tl.load` with explicit masks. This ensures correctness for arbitrary matrix sizes without requiring pre-processing or padding of the input tensors by the user, maintaining high throughput for edge cases.
- **in place accumulation strategy**：The accumulation loop uses `tl.dot` by passing the existing accumulator `mat_c_block` as the third argument. This enables in-place accumulation within the register file, reducing register pressure and minimizing the instruction count associated with moving intermediate results between different register banks or memory hierarchies.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/047-matmul_kernel.py
```

**坏实践（反例）**：行主序遍历 block + 对 M/N/K 三维整体 padding → 相邻核心访存跨行跨列、L2 命中率低，整体 padding 白算大量无效 tile → ✅ 正解即本条：`grouped_launch_diagonal` 对角遍历 + `dot_pad_only_k` 只 pad K 维；累加器不作为 `tl.dot` 第三参传入还会增加寄存器搬运

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_ascend_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically designed for the Ascend software stack, utilizing the `dl` dialect for hardware-specific compile hints. The `grouped_launch_diagonal` function is an external dependency that must be defined to handle the block-to-coordinate mapping.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](../compute-pipeline/exp_cube1_qk_gemm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gemm_scale_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](../compute-pipeline/exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic work stealing](../compute-pipeline/exp_grouped_gemm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: diagonal partitioning](../compute-pipeline/exp_matmul_ascend_diag_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: block tiling strategy](../compute-pipeline/exp_matmul_bias_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: diagonal tiling strategy](../compute-pipeline/exp_matmul_diag_tiling_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: hardware specific instruction](../compute-pipeline/exp_matmul_qn_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: micro tiling strategy](../compute-pipeline/exp_matmul_small_tile_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: block swizzling](../compute-pipeline/exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: small tile strategy](../compute-pipeline/exp_matmul_triton_base_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](../compute-pipeline/exp_matmul_vec_parallel_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](../compute-pipeline/exp_partial_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: fused rope computation](../compute-pipeline/exp_rope_matmul_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
