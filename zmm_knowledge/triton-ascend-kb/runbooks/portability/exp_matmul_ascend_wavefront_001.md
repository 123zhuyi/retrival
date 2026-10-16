---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比
description: 910b2/910b3/91093/91095 上大矩阵乘法用 grouped_launch_diagonal 波前遍历提升 cache 局部性，dl.compile_hint dot_pad_only_k 只 pad K 维，@triton.autotune 扫 tile 长宽比（128x256/256x128）与分组参数，声称收益 35%
tags: [跨芯片适配, matrix-operations, matmul]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/91093/91095 ｜ triton-ascend 版本 未标注（依赖 dl 模块 compile hint）｜ dtype 全部 ｜ shape 大规模矩阵（访存带宽为主要瓶颈时收益最大）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：对角/波前调度只对访存受限的大矩阵收益显著，计算受限或小矩阵场景与行主序无差；dl.compile_hint 为 Ascend 扩展，GPU 无对应物；autotune 的 tile 候选须满足 UB/寄存器约束，盲目加大 tile 会 spill；与 exp_matmul_ascend_001 同源，本卡多 autotune 维度；910C/310P 未验证（unverified)）

### 原则
- **wavefront scheduling**：The kernel utilizes `grouped_launch_diagonal` to map program IDs to matrix blocks. This diagonal (or wavefront) traversal order ensures that blocks processed consecutively share data rows from matrix A and columns from matrix B. This significantly increases data locality in the cache (L2/SRAM), reducing redundant global memory accesses compared to a row-major or column-major traversal.
- **hardware specific alignment**：The application of `dl.compile_hint(..., "dot_pad_only_k")` is a critical hardware adaptation. It instructs the backend compiler to pad the inputB-dimension of the data blocks specifically for the dot product operation. This ensures that the data layout aligns perfectly with the requirements of the hardware's matrix units (e.g., the Cube unit on Ascend processors) without the overhead of padding the entire global tensor, maximizing compute throughput.
- **autotuning block shapes**：The `@triton.autotune` configuration sweeps through different tile aspect ratios (128x256 vs 256x128) and varying `BLOCK_TRESHHOLD` values. This allows the kernel to dynamically select the most efficient tile size and wavefront grouping parameter based on the specific N and inputB dimensions of the input matrices, balancing register pressure and instruction-level parallelism.
- **static multi core distribution**：The loop `for block_idx in range(pid, NUM_BLOCKS, NUM_CORES)` implements a static block-cyclic distribution of work across the available cores. This minimizes scheduling overhead and ensures load balancing, especially when the total number of blocks is not perfectly divisible by the number of cores.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/074-matmul_kernel.py
```

**坏实践（反例）**：行主序/列主序映射 program id 到 block → 同时调度的核心访问的 A 行、B 列互不重叠，L2/SRAM 冗余加载严重 → ✅ 正解即本条：对角波前遍历让并发 block 共享 A 行与 B 列；再配 `dot_pad_only_k` 与 tile 长宽比 autotune

**已知实例**：matmul §raw (`experiences/matmul/exp_matmul_ascend_wavefront_001.yaml`)
- 声称性能收益：35%
- 备注：This implementation relies on the `dl` module (likely a deep learning library extension for Ascend) for compile hints. The diagonal scheduling strategy is most effective for large-scale matrix multiplications where memory bandwidth is the primary bottleneck.

<!-- okf:related:start -->

# 相关

- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 同属优化知识，主题相关
- 相关主题: [累加/原地修改算子 autotune 必须每次重置输出 buffer](../autotuning/autotune-debug.md) — 同属优化知识，主题相关
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](../autotuning/tiling-candidate-compile-cost-gate.md) — autotune tile 扫描实践需过本卡编译成本闸
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](../autotuning/triton-ascend-microkernel-autotune-noise.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: tiling strategy](../compute-pipeline/exp_cube1_qk_gemm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_gemm_scale_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: async pipeline scheduling](../compute-pipeline/exp_fused_matmul_norm_sigmoid_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rms_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rmsnorm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: operator fusion](../compute-pipeline/exp_fused_matmul_rope_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic work stealing](../compute-pipeline/exp_grouped_gemm_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_gemm_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_gemm_bnmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: dynamic load balancing](../compute-pipeline/exp_grouped_matmul_bkmajor_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: block tiling strategy](../compute-pipeline/exp_matmul_bias_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Scheduling Optimization: block swizzling](../compute-pipeline/exp_matmul_swizzle_hardware_hint_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: vector parallelism strategy](../compute-pipeline/exp_matmul_vec_parallel_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: kernel fusion](../compute-pipeline/exp_mhc_fused_gemm_sqrsum_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: grid stride scheduling](../compute-pipeline/exp_partial_matmul_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Compute Optimization: fused rope computation](../compute-pipeline/exp_rope_matmul_fused_001.md) — 主题相关：共享 matmul, matrix-operations
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
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
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matmul, matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
