---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局
description: 910b2/910b3/91093/91095 上 RoPE 类 kernel 按 BLOCK_Q/blockK 分块保持工作集在 UB 内；复数乘法用 tl.math.fma 融合乘加减指令数；reshape can_reorder=True + tl.interleave 处理交错复数布局，避免显式转置/gather，声称收益 25%
tags: [跨芯片适配, matrix-operations, rope]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/91093/91095 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 浮点（复数实虚部）｜ shape 头维分块（BLOCK_Q × blockK）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：假定频率输入输出为交错复数布局（real, imag 相邻），非交错布局时 tl.interleave/can_reorder 不适用需先转置；工作集本已小于 UB 时分块收益消失；fma 与分离乘加的精度差异（单舍入 vs 双舍入）在严格数值比对场景需评估；910C/310P 未验证（unverified)）

### 原则
- **ub tiling strategy**：The kernel iterates over query and key heads in chunks defined by BLOCK_Q and blockK. This tiling strategy is designed to keep the active working set within the NPU's Unified Buffer (UB), minimizing high-latency data transfers between Global Memory (GM) and UB.
- **fma instruction utilization**：The complex multiplication arithmetic is implemented using tl.math.fma (Fused Multiply-Add). This leverages the NPU's pipeline capabilities to compute the real and imaginary components (ac - bd, ad + bc) efficiently, reducing the total number of instructions compared to discrete multiplication and addition.
- **interleaved layout optimization**：By using can_reorder=True in reshape operations and tl.interleave for the output, the kernel allows the compiler to handle the interleaved complex data layout (real, imag) optimally. This avoids costly explicit transposition or gather/scatter operations and aligns memory access patterns with the NPU's vectorization units.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/098-_triton_llama4_rope_npu.py
```

**坏实践（反例）**：复数乘法写成离散的两次乘加（`ac - bd`、`ad + bc` 分开算）并对交错布局做显式转置/gather → 指令数翻倍且访存模式与向量化单元不对齐 → ✅ 正解即本条：`tl.math.fma` 融合乘加 + `tl.interleave`/`can_reorder=True` 让编译器处理交错布局

**已知实例**：rope §raw (`experiences/rope/exp_rope_llama4_npu_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically tuned for the Ascend NPU memory hierarchy. It assumes an interleaved complex layout for the frequency inputs and outputs.

<!-- okf:related:start -->

# 相关

- 实践案例: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 相关主题: [数学归一化与算法前置条件必须显式处理（FFT 归一化/方向交替/模归约）](../algorithm-numerics/math-normalize-precondition.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: dynamic autotuning](../compute-pipeline/exp_addmm_ascend_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Scheduling Optimization: dual task pipelining](../compute-pipeline/exp_attn_ascend_dual_pipeline_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [Compute Optimization: blocking strategy](../compute-pipeline/exp_attn_qk_cube_001.md) — 主题相关：共享 matrix-operations
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关
- 相关主题: [pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)](../memory-sram/pergroup-blockk-align.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend strided-2/interleave 访存契约（三个正确性缺陷 + 寄存器重排正解）](../memory-sram/triton-ascend-strided2-interleave-contract.md) — 主题相关：共享 rope
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
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
