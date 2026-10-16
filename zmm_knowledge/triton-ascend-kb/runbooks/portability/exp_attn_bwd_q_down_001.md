---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存
description: 910b2/910b3 上用 tile_mix_vector_loop/tile_mix_cube_loop 编排指令流水线（Cube 跑 GEMM、Vector 跑逐元素），enable_hivm_auto_cv_balance 交给编译器动态均衡，set_workspace_multibuffer 2 双缓冲掩盖 GM 延迟；长序列按 SEQ_LEN=S//2 拆 down kernel 适配 L0C/L1，声称收益 35%
tags: [跨芯片适配, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（依赖 CANN HIVM 扩展）｜ dtype 混合精度（FP32 累加）｜ shape 长序列（SEQ_LEN = S // 2 分半）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：tile_mix_*/multibuffer/hivm_auto_cv_balance 为 CANN 专用扩展，GPU Triton 不可用；序列分半的 down kernel 必须与对应 up kernel 配对使用，单独部署只覆盖一半序列；片上内存（L0C/L1）足以容纳完整序列时无需拆分；910C/310P 未验证（unverified)）

### 原则
- **hardware unit pipelining**：Utilizes `tile_mix_vector_loop` and `tile_mix_cube_loop` configuration parameters to explicitly orchestrate the instruction pipeline. This ensures the AI Cube unit handles matrix multiplications (GEMM) while the Vector unit handles element-wise operations (masking, arithmetic) concurrently, maximizing the throughput of the Ascend AI Core.
- **compiler auto balancing**：Enables `enable_hivm_auto_cv_balance` to delegate instruction scheduling to the HiAI compiler. This allows the runtime to dynamically balance the workload between the Cube and Vector pipelines based on actual data dependencies, effectively reducing pipeline stalls.
- **multibuffering strategy**：Activates `multibuffer` with a workspace size of 2 (`set_workspace_multibuffer`). This implements double-buffering for data loading (specifically for inputB and inputC tensors in the micro-kernel), overlapping global memory access latency with the compute cycles of the previous block.
- **sequence partitioning**：The kernel logic processes `SEQ_LEN = S // 2`, indicating a "down" partitioning strategy. This splits the sequence length reduction across multiple kernel launches to better fit within the limited on-chip memory (L0C/L1) and register file constraints of the NPU when dealing with long contexts.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/263-kernel_sda_bwd_q_down.py
```

**坏实践（反例）**：长序列反向不在序列维拆分、单 kernel 硬扛完整 S → L0C/L1/寄存器溢出导致编译失败或频繁 spill → ✅ 正解即本条：SEQ_LEN = S // 2 拆 down kernel 与 up kernel 配对，配合 tile_mix 流水线编排与 multibuffer 双缓冲

**已知实例**：attention §raw (`experiences/attention/exp_attn_bwd_q_down_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is specifically tailored for the Ascend 910B series architecture using HiAI VM extensions. It is designed to work in conjunction with an "up" kernel to cover the full sequence length.

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: attention sinks mechanism](../compute-pipeline/exp_attn_sinks_prefill_001.md) — 主题相关：共享 attention, matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: grouped scheduling](../compute-pipeline/exp_bmm_grouped_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](exp_gqa_paged_npu_001.md) — 主题相关：共享 attention, 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配

<!-- okf:related:end -->
