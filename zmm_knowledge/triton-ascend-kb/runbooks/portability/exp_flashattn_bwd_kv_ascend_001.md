---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半
description: 910b2/910b3 上反向 attention KV kernel 用 enable_hivm_auto_cv_balance + set_workspace_multibuffer 2 + tile_mix loop=2 均衡 Cube/Vector 流水线，BLOCK_R=128/BLOCK_C=64 非对称 tile 匹配 KV 访存粒度，SEQ_LEN=S//2 分半减少因果掩码开销，声称收益 30%
tags: [跨芯片适配, matrix-operations, attention]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（依赖 HiAI VM 扩展，须确认后端支持这些 flag）｜ dtype 混合精度 ｜ shape 长序列（SEQ_LEN = S // 2）、KV 非方形访存
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：enable_hivm_auto_cv_balance / set_workspace_multibuffer / tile_mix_* 为 Ascend HiAI VM 专用 flag，GPU Triton 及不支持的后端版本会报错或静默失效；BLOCK_R=128/BLOCK_C=64 针对 KV 非方形访存调优，方形或其他访存比例需重新调；序列分半仅在因果掩码开销显著的反向场景有收益；910C/310P 未验证（unverified)）

### 原则
- **hardware pipeline balancing**：Enabled `enable_hivm_auto_cv_balance` to allow the compiler to automatically balance the workload between the Cube (matrix compute) and Vector units, preventing pipeline stalls on Ascend NPUs.
- **multi buffering**：Configured `multibuffer True` with `set_workspace_multibuffer 2` to overlap data transfer of inputB/inputC tiles with the computation of the micro-kernel, effectively hiding memory latency.
- **hybrid loop unrolling**：Set `tile_mix_vector_loop` and `tile_mix_cube_loop` to 2, optimizing the instruction issue rate by interleaving vector and cube operations to maximize hardware throughput.
- **asymmetric tiling**：Utilized `BLOCK_R=128` and `BLOCK_C=64` to match the NPU's memory access granularity, ensuring efficient loading of the KV cache blocks which are often accessed in non-square patterns during the backward pass.
- **sequence partitioning**：Implemented a sequence splitting strategy (`SEQ_LEN = S // 2`) to optimize the iteration space for the backward pass, reducing the overhead of causal mask checks within the innermost loops.

### 代码/骨架出处
- `TritonPath`:
```python
kernel_sda_bwd_kv_right
```

**坏实践（反例）**：KV 反向沿用前向的方形 tile（如 128×128）→ KV cache 非方形访存模式下 tile 与硬件访存粒度不匹配，且最内层循环逐次做因果掩码判断 → ✅ 正解即本条：非对称 BLOCK_R=128/BLOCK_C=64 + 序列分半（SEQ_LEN = S // 2）+ CV 均衡与双缓冲 flag

**已知实例**：attention §raw (`experiences/attention/exp_flashattn_bwd_kv_ascend_001.yaml`)
- 声称性能收益：30%
- 备注：This configuration relies on specific Ascend NPU features (HiAI VM). Ensure the Triton backend supports these flags.

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
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 attention, matrix-operations, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
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
