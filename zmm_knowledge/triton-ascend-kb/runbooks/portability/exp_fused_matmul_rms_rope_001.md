---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖
description: 910b2/910b3 上把计算密集的 MatMul 派给 Cube 单元、访存密集的归一化与旋转位置编码派给 Vector 单元，用 dl.async_task 异步调度使向量操作隐藏在 GEMM 之后，三算子单 kernel 融合减少启动开销，声称收益 25%
tags: [跨芯片适配, matrix-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（需支持 dl.async_task 异步调度）｜ dtype 混合精度 ｜ shape 任意（nd）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：要求硬件支持 Cube/Vector 异步任务调度（910B 系列）；被融合的向量侧操作必须与 MatMul 无数据依赖（或可重排），存在依赖时异步化会算错；向量侧计算量远大于 GEMM 时掩盖不住，融合收益退化为仅省 kernel 启动开销；910C/310P 未验证（unverified)）

### 原则
- **heterogeneous parallelism**：The kernel explicitly leverages the NPU's heterogeneous architecture by dispatching the compute-intensive Matrix Multiplication to the `cube` unit and the memory-intensive RMS Normalization and Rotary Position Embedding to the `vector` unit. This ensures that different hardware units are utilized simultaneously rather than sequentially.
- **async task scheduling**：By using `dl.async_task`, the runtime is instructed to execute the MatMul and the vector operations (RMS Norm + RoPE) asynchronously. This allows the vector unit to process the normalization and embedding logic in parallel with the matrix calculations on the cube unit, effectively hiding the latency of the vector operations behind the heavy GEMM computation.
- **kernel fusion graph**：The code fuses three distinct operations (MatMul, RMS Norm, RoPE) into a single kernel launch context. This reduces the overhead of multiple kernel launches and improves data locality, particularly for the `KV` tensor which is processed sequentially by the vector kernels.

### 代码/骨架出处
- `TritonPath`:
```python
fused_matmul_and_rms_norm_and_rotary_pos_emb_kernel
```

**坏实践（反例）**：MatMul、归一化、位置编码各自独立 kernel 顺序执行 → 三次 kernel 启动开销 + Vector 单元在 GEMM 期间空转 → ✅ 正解即本条：单 kernel 融合 + `dl.async_task` 让 Vector 操作与 Cube GEMM 并行；注意向量操作与 MatMul 有数据依赖时不能异步化

**已知实例**：matmul §raw (`experiences/matmul/exp_fused_matmul_rms_rope_001.yaml`)
- 声称性能收益：25%
- 备注：This optimization is highly effective for Transformer inference/decoding stages where MatMul is followed by normalization and RoPE. It requires hardware support for asynchronous task scheduling between Cube and Vector units (e.g., Ascend 910B).

<!-- okf:related:start -->

# 相关

- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](../compute-pipeline/async-task-cube-vector.md) — 同属优化知识，主题相关
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
- 相关主题: [Scheduling Optimization: micro tiling strategy](../compute-pipeline/exp_matmul_small_tile_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: block swizzling](../compute-pipeline/exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: small tile strategy](../compute-pipeline/exp_matmul_triton_base_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](../compute-pipeline/exp_matmul_vec_parallel_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: pid grouping strategy](../compute-pipeline/exp_mm_grouped_reorder_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](../compute-pipeline/exp_partial_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: fused rope computation](../compute-pipeline/exp_rope_matmul_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
