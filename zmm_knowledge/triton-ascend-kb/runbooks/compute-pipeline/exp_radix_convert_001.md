---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: 'Compute Optimization: compile time dispatch'
description: 'misc 场景下，Compute Optimization: compile time dispatch，收益 20%'
tags: [AICore 单元利用/Scalar 削减, element-wise-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Compute Optimization: compile time dispatch

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3, 91093, 91095；失效边界：仅适用于 radix 类排序前置的有符号→偏移二进制转换；dtype 种类多时 constexpr 分发的编译产物膨胀；输入已是 unsigned 或不需要保序时本条无意义）

### 原则
- **compile time dispatch**：The kernel utilizes `tl.constexpr` to evaluate data type conditions (int8, int16, etc.) during compilation rather than at runtime. This eliminates branch divergence within the GPU/NPU warps, ensuring that the generated machine code is specialized and efficient for the specific input tensor type.
- **sign bit flipping algorithm**：Implements a highly optimized method to flip the Most Significant Bit (MSB) using bitwise shifts (`>>`) and masks (`&`). This converts Two's Complement signed integers into Offset Binary (unsigned) format, which is a critical step for maintaining lexicographical order when performing Radix Sort on signed data using unsigned comparators.
- **hardware instruction mapping**：The operations used (bitwise AND, OR, SHIFT) map directly to single-cycle integer arithmetic instructions on the hardware, avoiding the overhead of complex arithmetic or floating-point conversion logic.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/135-radix_type_convert.py
```

**坏实践（反例）**：dtype 分支用运行时 if 判断（int8/int16/...）→ warp 内分支分歧 + 通用代码路径 → ✅ 正解即本条：`tl.constexpr` 编译期分发 + MSB 翻转移位/掩码映射到单周期整数指令。

**已知实例**：misc §raw (`experiences/misc/exp_radix_convert_001.yaml`)
- 声称性能收益：20%
- 备注：This function is essential for preprocessing signed integer data before passing it into a Radix Sort kernel that operates on unsigned integers. It preserves the relative ordering of negative and positive numbers.

<!-- okf:related:start -->

# 相关

- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — host 成本模型分档 dispatch 与 compile time dispatch 同族
- 相关主题: [Moe Dispatch Skeleton](../../reference/skeleton-moe_dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](../autotuning/tiling-candidate-compile-cost-gate.md) — host 公式分档与 compile time dispatch 同属确定性分派
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](../memory-sram/exp_cache_store_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: compile time branching](../memory-sram/exp_causal_conv1d_states_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: operator fusion](../memory-sram/exp_fuse_tlx_001.md) — 主题相关：共享 misc
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 主题相关：共享 misc
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）](../meta-skills/precompute-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](chunk-recurrence-pingpong-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr tiling](exp_block_inverse_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: gemm based histogram](exp_digit_hist_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: grid stride loop](exp_element_mul_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: compilation specialization control](exp_expand_serial_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: operator fusion](exp_fused_index_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: deep operator fusion](exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](exp_grpo_loss_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: register accumulation](exp_hist_count_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: diagonal traversal pattern](exp_launch_diagonal_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: persistent kernel grid stride loop](exp_linear_persistent_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: Operator Fusion](exp_mhc_post_fusion_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized binary search](exp_multinomial_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: high precision accumulation](exp_recompute_w_u_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized load](exp_reject_sampler_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Scheduling Optimization: vectorized global load](exp_rejection_sample_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: block local sorting](exp_shuffle_blockwise_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](exp_solve_tril_004.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: padding strategy](exp_sort_bitonic_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: scalar loop vectorization](exp_spec_decode_verify_scalar.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: vectorized probability calculation](exp_top_p_sampling_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: kernel fusion](exp_topp_filter_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc
- 相关主题: [Compute Optimization: micro kernel tiling](exp_tril_solve_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减, misc

<!-- okf:related:end -->
