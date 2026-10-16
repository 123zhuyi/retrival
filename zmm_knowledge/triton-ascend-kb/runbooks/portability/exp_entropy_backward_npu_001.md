---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile
description: 910b2/910b3/91093/91095 上标签索引做相等比较前先转 int32 避免 int64 标量降级；GROUP_SIZE_M=4 分组调度提升 L2 局部性；BLOCK_SIZE_K=512 让 Cube 单元数据量充足，声称收益 25%
tags: [跨芯片适配, matrix-operations, loss]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3/91093/91095 ｜ triton-ascend 版本 未标注（按验证版本）｜ dtype 索引 int32（比较前转换）｜ shape 大 K（BLOCK_SIZE_K=512）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：标签/索引值域超过 int32 范围时不得转换（溢出）；GROUP_SIZE_M 分组只对多核共享输入行的场景有 L2 收益，单核或小 batch 无效；BLOCK_SIZE_K=512 以 UB 容量放得下为前提，dtype 位宽翻倍时须等比缩小；910C/310P 未验证（unverified)）

### 原则
- **npu int32 casting**：Converts label indices to int32 before equality comparison to prevent the NPU from degrading to scalar operations, a critical optimization for Ascend hardware where int64 logic is inefficient.
- **grouped scheduling**：Implements GROUP_SIZE_M=4 to improve L2 cache locality for the hidden tensor, ensuring that consecutive program IDs process the same rows of the input matrix to reduce memory access overhead.
- **large k tiling**：Sets BLOCK_SIZE_K=512 to maximize the utilization of the AI Core's Cube unit, ensuring sufficient data volume for pipelining during the matrix multiplication phase.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/205-efficient_entropy_backward_kernel_general_d_logits_split_N.py
```

**坏实践（反例）**：直接用 int64 标签索引做相等比较（`labels == vocab_idx`，labels 保持 int64）→ Ascend 上 int64 逻辑运算标量降级，逐元素串行 → ✅ 正解即本条：比较前 `labels.to(tl.int32)`；注意本卡只覆盖 EQ 比较场景，LT/GT 等 i32 比较仍降级（见 exp_ascend_int_compare_scalar_001）

**已知实例**：loss §raw (`experiences/loss/exp_entropy_backward_npu_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is specifically optimized for the Ascend NPU architecture. The int32 casting logic is essential to avoid severe performance penalties associated with int64 comparisons on these devices.

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeComputeExpertTokens (MoE expert token count via binary search) on Ascend NPU](../../ops/moe-compute-expert-tokens.md) — 同属优化知识，主题相关
- 开发指南: [Loss 函数算子优化](../../reference/op-class-loss.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_bwd_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_celoss_smooth_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_celoss_smooth_bwd_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_crossentropy_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: grid stride scheduling](../compute-pipeline/exp_crossentropy_backward_001.md) — 主题相关：共享 loss
- 相关主题: [Scheduling Optimization: kernel fusion and recomputation](../compute-pipeline/exp_entropy_backward_001.md) — 主题相关：共享 loss, matrix-operations
- 相关主题: [Scheduling Optimization: persistent kernel grid stride](../compute-pipeline/exp_entropy_backward_d_weight_001.md) — 主题相关：共享 loss, matrix-operations
- 相关主题: [Scheduling Optimization: grid stride scheduling](../compute-pipeline/exp_entropy_backward_fused_001.md) — 主题相关：共享 loss, matrix-operations
- 相关主题: [Compute Optimization: numerical stability reduction](../compute-pipeline/exp_entropy_epilogue_001.md) — 主题相关：共享 loss
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_entropy_001.md) — 主题相关：共享 loss, matrix-operations
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 相关主题: [kernel 内索引必须转 int32，避免 int64 降级](../compute-pipeline/int32-index-cast.md) — 同属优化知识，主题相关
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [核心数在 __init__ 获取一次，按算子类型选 VEC/CUBE 核数](exp_ascend_core_num_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [大 shape 用交错循环：grid 固定为核心数，kernel 内按步长遍历](exp_ascend_grid_large_shape_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [地址计算全程保持 int32，避免 int64 标量降级](exp_ascend_int64_address_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [整数比较标量降级：比较操作数转 fp32 或编译期消除](exp_ascend_int_compare_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [并行任务分配首选 for i in range(pid, total, core_num) 交错模式](exp_ascend_parallel_pattern_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [BLOCK_SIZE <65536 且子块内存 ≤L1 缓存，超限按维度多次嵌套切分](exp_ascend_split_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [GQA 分页 KV：tl.parallel bind_sub_block 双向量核并行 + KV 按组复用 + block_table 遍历](exp_gqa_paged_npu_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历](exp_sda_bwd_q_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配

<!-- okf:related:end -->
