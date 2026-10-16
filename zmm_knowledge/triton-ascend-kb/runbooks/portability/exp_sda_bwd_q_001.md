---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.3
title: 反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历
description: 910b2/910b3 上反向 Q kernel 用 tile_mix loop=2 + enable_hivm_auto_cv_balance 均衡 Cube/Vector，multibuffer=2 双缓冲；掩码用代数写法 (mask.to(fp32)-1.0)*1e6 替代 if-else 避免分支分化；64×128 tile 按对角-下三角-前列结构化遍历，声称收益 35%
tags: [跨芯片适配, matrix-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 反向 Q kernel：CV 均衡 + 双缓冲 + 代数掩码（(mask-1)*1e6）+ 64×128 三角遍历

【优化维度】：跨芯片适配
【适用条件】：平台 910b2/910b3 ｜ triton-ascend 版本 未标注（依赖 NPU 专用 flag）｜ dtype bf16 存储 + fp32 累加（HIGH_TYPE）｜ shape 稀疏/滑窗类注意力（SEQ_LEN = S // 2）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：代数掩码 (mask-1)*1e6 要求后续接 softmax 类归一化使 -1e6 趋零，直接对掩码结果做算术的场景不适用；SEQ_LEN=S//2 的半长索引只对特定稀疏/滑窗模式成立，全量注意力误用会漏算；三角结构化遍历针对反向访存模式调优，前向或其他模式需重调；910C/310P 未验证（unverified)）

### 原则
- **ascend cube vector balance**：The configuration utilizes `tile_mix_vector_loop` and `tile_mix_cube_loop` set to 2, alongside `enable_hivm_auto_cv_balance`. This explicitly balances the instruction dispatch between the AI Cube unit (for matrix multiplication) and the Vector unit (for element-wise operations like softmax and masking) on Ascend NPUs, maximizing pipeline utilization.
- **multibuffer prefetch**：The `multibuffer True` and `set_workspace_multibuffer 2` settings enable double buffering. This allows the hardware to prefetch data for the next tile while computing the current tile, effectively hiding global memory access latency.
- **algebraic masking**：Instead of using conditional branching (if-else) which causes thread divergence, the kernel implements masking via `(block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6`. This algebraic approach converts the mask into a large negative number added to the logits, ensuring SIMD execution efficiency.
- **structured tiling strategy**：The kernel employs a 64x128 tiling scheme (`BLOCK_R=64`, `BLOCK_C=128`) and iterates over the sequence dimension in a structured order (diagonal, lower triangular, then previous columns). This specific pattern optimizes for the memory access patterns of the backward pass in sparse or sliding-window attention mechanisms (implied by `SEQ_LEN = S // 2`).

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/262-kernel_sda_bwd_q_up.py
```

**坏实践（反例）**：掩码用 if-else 条件分支实现 → warp/向量化执行内分支分化，SIMD 效率骤降 → ✅ 正解即本条：代数掩码 `(block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6` 把掩码变成加到 logits 上的大负数，纯向量指令无分支

**已知实例**：misc §raw (`experiences/misc/exp_sda_bwd_q_001.yaml`)
- 声称性能收益：35%
- 备注：This kernel is highly optimized for Ascend 910B series hardware using specific NPU intrinsics flags. The logic implies a specific attention pattern (e.g., Sliding Window or ALiBi) where the sequence length is effectively halved for internal indexing.

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_element_mul_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_index_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](../compute-pipeline/exp_launch_diagonal_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](../compute-pipeline/exp_linear_persistent_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: Operator Fusion](../compute-pipeline/exp_mhc_post_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: micro kernel tiling](../compute-pipeline/exp_tril_solve_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [整数比较/加法必须转 float32，避免 Vector 标量降级](../compute-pipeline/int-compare-cast-float32.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: kernel fusion](../memory-sram/exp_logits_penalty_fusion_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](../memory-sram/exp_outer_product_001.md) — 主题相关：共享 matrix-operations, misc
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 同属优化知识，主题相关
- 相关主题: [内存访问三查：mask 防越界、stride 匹配布局、索引上限判断](../meta-skills/exp_debug_001.md) — 同属优化知识，主题相关
- 相关主题: [控制流避坑：禁 return/break/continue，复杂条件分步 mask 组合](../meta-skills/exp_debug_002.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](exp_ascend_blockptr_boundary_scalar_001.md) — 主题相关：共享 跨芯片适配
- 相关主题: [Cube/Vector 流水线编排 + 序列分半（down kernel）适配片上内存](exp_attn_bwd_q_down_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [Ascend 专用 tiling 配置：tile_mix cube/vector loop + multibuffer + HIVM CV 自动均衡](exp_attn_sda_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [标签索引转 int32 比较 + GROUP_SIZE_M 分组调度 + 大 K tile](exp_entropy_backward_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV kernel：HIVM CV 均衡 + 双缓冲 + 非对称 tile（128×64）+ 序列分半](exp_flashattn_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [异构并行融合：MatMul 派 Cube、归一化+位置编码派 Vector，dl.async_task 异步掩盖](exp_fused_matmul_rms_rope_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [MatMul 调度：对角遍历 grouped_launch + dot_pad_only_k hint + 寄存器原位累加](exp_matmul_ascend_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](exp_matmul_ascend_wavefront_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [UB tiling（BLOCK_Q×blockK 分块）+ tl.math.fma 复数乘法 + tl.interleave 交错布局](exp_rope_llama4_npu_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [dot_pad_only_k hint + 对角 block 调度 + batch/head 循环内置融合](exp_rope_matmul_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [反向 KV 梯度 kernel：HIVM CV 均衡 + tile_mix=2 + 双缓冲 + 256×64 tile + bf16 存储 fp32 累加](exp_sda_bwd_kv_ascend_001.md) — 主题相关：共享 matrix-operations, misc, 跨芯片适配
- 相关主题: [前向 SDA kernel：CV 均衡 + 双缓冲 + bf16 乘 fp32 累加 + 64×256 tile 分相处理](exp_sda_fwd_ascend_001.md) — 主题相关：共享 matrix-operations, misc, 跨芯片适配
- 相关主题: [稀疏注意力 prefill：tle.index_select_simd 按需取 Top-K + cube1 加速 + 工作集 tile=64](exp_sparse_flash_attn_prefill_001.md) — 主题相关：共享 matrix-operations, 跨芯片适配
- 相关主题: [Triton-Ascend mask 纪律与编译期陷阱（fp32 mask 污染 / stride-0 memref / 禁 % 运算符 / 2D mask store 丢格）](triton-ascend-mask-codegen-traps.md) — 代数掩码写法与 mask 纪律相关

<!-- okf:related:end -->
