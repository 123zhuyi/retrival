---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理
description: 910b2/910b3 上前向滑窗/局部注意力 kernel 用 enable_hivm_auto_cv_balance + tile_mix loop 均衡 Cube/Vector，multibuffer=2 双缓冲，bf16 矩阵乘 + fp32 在线 softmax 累加，BLOCK_R=64/BLOCK_C=256 且序列分半（S//2）分相处理局部/因果/全局模式，声称收益 35%
tags: [跨芯片适配, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（依赖 HiVM 专用 flag）｜ dtype bf16 乘 + fp32 累加（LOW_TYPE/HIGH_TYPE）｜ shape 滑窗/局部注意力（SEQ_LEN = S // 2 分相）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：HiVM flag 为 Ascend 专用，GPU Triton 不可用；S//2 分相逻辑只对局部/因果/全局混合的特定注意力模式成立，全量注意力误用会漏算；BLOCK_R=64/BLOCK_C=256 按 UB/L0C 容量调优，换机型或 dtype 需重调；910C/310P 未验证（unverified)）

### 原则
- **hardware unit balancing**：Utilizes `enable_hivm_auto_cv_balance`, `tile_mix_vector_loop`, and `tile_mix_cube_loop` to optimally distribute instructions between the Ascend Cube unit (for matrix multiplication in QK^T and Softmax(inputC)) and the Vector unit (for softmax, masking, and scaling), preventing pipeline bubbles.
- **multibuffer prefetching**：Enables `multibuffer` with `set_workspace_multibuffer 2` to overlap the loading of inputB and inputC tiles with the computation of the current inputA tile, effectively hiding global memory access latency.
- **hybrid precision accumulation**：Performs matrix multiplication in `bf16` (LOW_TYPE) to maximize Cube unit throughput, while maintaining the accumulator `block_o`, `block_m`, and `block_l` in `fp32` (HIGH_TYPE) to preserve numerical stability during the online softmax reduction.
- **structured tiling strategy**：Implements a specialized tiling scheme with `BLOCK_R=64` and `BLOCK_C=256` tailored for Sliding Decoding Attention (SDA). The logic splits the sequence length (`S // 2`) to efficiently handle local, causal, and global attention patterns in distinct phases, minimizing redundant memory loads.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/259-kernel_sda_fwd_up.py
```

**坏实践（反例）**：在线 softmax 的累加器（block_o/block_m/block_l）也用 bf16 → Log-Sum-Exp 增量更新中舍入误差逐块累积，长序列下精度明显漂移 → ✅ 正解即本条：矩阵乘 bf16 保吞吐、累加器 fp32 保稳定；不序列分相而统一处理则局部/全局模式互相干扰、冗余加载

**已知实例**：misc §raw (`experiences/misc/exp_sda_fwd_ascend_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically optimized for Ascend 910B series hardware using HiVM specific flags. The logic assumes a specific attention pattern where sequence processing is split (SEQ_LEN = S // 2).

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
- 相关主题: [Memory Access Optimization: kernel fusion](../memory-sram/exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](../memory-sram/exp_outer_product_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
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
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](exp_gqa_paged_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, misc, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, misc, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
