---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合
description: 910b2/910b3/91093/91095 上矩阵乘用 dl.compile_hint "dot_pad_only_k" 只 pad K 维对齐 Cube 单元，grouped_launch_diagonal 对角遍历提升 L2 命中，BATCH/HEADS 循环内置 kernel 体减少启动开销，声称收益 35%
tags: [跨芯片适配, matrix-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/91093/91095 ｜ triton-ascend 版本 未标注（依赖 dl 方言）｜ dtype 全部 ｜ shape 带 batch/head 维的矩阵乘
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：dl.compile_hint 为 Ascend 扩展，GPU Triton 不可用；batch/head 循环内置要求单次迭代工作集不超 UB/寄存器，否则 spill 抵消融合收益；与 exp_matmul_ascend_001/exp_matmul_ascend_wavefront_001 共享对角调度与 dot hint 主题，本卡多 batch/head 融合维度；910C/310P 未验证（unverified)）

### 原则
- **hardware specific dot hint**：Utilizes dl.compile_hint with the argument "dot_pad_only_k" for both input matrices. This instructs the compiler backend to specifically pad the inputB-dimension to align with the hardware's Cube unit requirements (e.g., 16-byte alignment) without padding the M and N dimensions. This significantly reduces unnecessary data loading and computation compared to full-block padding.
- **diagonal block scheduling**：Implements a diagonal traversal strategy for processing matrix blocks via the grouped_launch_diagonal function. This pattern optimizes L2 cache hit rates by ensuring that adjacent blocks in the computation wavefront share rows of matrix A and columns of matrix B, thereby minimizing global memory bandwidth pressure.
- **fused batch head processing**：Integrates loops for BATCH and HEADS directly inside the kernel body rather than launching separate kernels for each batch/head. This reduces kernel launch overhead and allows for better pipelining of instructions across the different data slices.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/057-_single_rope_matmul_kernel.py
```

**坏实践（反例）**：每个 batch/head 单独 launch 一个 kernel + 对 M/N/K 整体 padding → 启动开销随 batch×heads 线性增长，整体 padding 白算无效 tile → ✅ 正解即本条：BATCH/HEADS 循环内置 kernel 体 + `dot_pad_only_k` 只 pad K 维

**已知实例**：matmul §raw (`experiences/matmul/exp_rope_matmul_001.yaml`)
- 声称性能收益：35%
- 备注：The 'dl' module usage indicates this is optimized for specific NPU hardware (likely Ascend series). The "dot_pad_only_k" hint is critical for achieving peak performance on the Cube units.

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
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](../compute-pipeline/exp_partial_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Memory Access Optimization: paged kv cache with gqa head tiling](../memory-sram/exp_paged_gqa_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Memory Access Optimization: head vectorization](../memory-sram/exp_paged_kv_cache_001.md) — 同属优化知识，主题相关
- 相关主题: [layout (B,T,H,D)→(B,H,T,D) 消除跨 Head 离散访存](../memory-sram/layout-bthd-to-bhtd.md) — 同属优化知识，主题相关
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
