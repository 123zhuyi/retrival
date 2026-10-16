---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: fused multi-tensor split and reshape"
description: "misc 场景下，Memory Access Optimization: fused multi-tensor split and reshape，收益 25%"
tags: [访存效率, concatenation-operations, misc]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: fused multi-tensor split and reshape

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，各切分段边界为编译期常量（constexpr）；失效边界：输入 buffer 非连续排布时需先做布局变换，预处理开销抵消融合收益；切分段数很少（如仅 2 段）时多 kernel launch 开销本身很小，融合收益稀释）

### 原则
- **kernel fusion**：Fuses the splitting and reshaping of inputA, inputB, inputC, Z, B, and A tensors from two large fused input buffers (`mixed_qkvz`, `mixed_ba`) into five distinct output buffers. This eliminates multiple kernel launch overheads and reduces global memory read/write traffic compared to executing separate split kernels.
- **static loop unrolling**：Utilizes `tl.static_range` for the loops handling the B and A tensor splits. Since the loop bounds are defined as `constexpr` (compile-time constants), the compiler fully unrolls these loops. This removes dynamic loop control overhead (branching) and allows the compiler to optimize memory instructions more effectively.
- **vectorized memory access**：Leverages `tl.load` and `tl.store` with `tl.arange` to process contiguous memory segments. This ensures GM→UB accesses are contiguous and vectorized, maximizing the utilization of the memory bus bandwidth on the Ascend hardware.
- **compile time constants**：Defines dimensions like `QKVZ_DIM_T`, `BA_DIM_T`, and slice boundaries (`q_end`, `k_end`, etc.) as `tl.constexpr`. This enables the compiler to perform aggressive constant folding and address calculation optimization during the compilation phase rather than at runtime.

### 代码/骨架出处
- `TritonPath`:
```python
raw_experiences/extracted_kernels/157-fused_qkvzba_split_reshape_cat_kernel.py
```

**坏实践（反例）**：把两段大 buffer 的切分/reshape 拆成多个独立 kernel，每路输出一个 kernel
```python
for seg in segments:                        # 每段一个独立 kernel
    dst = tl.load(src_ptr + seg.start + tl.arange(0, BLOCK))
    tl.store(out_ptrs[seg.id] + off, dst)   # 同一输入 buffer 被多次启动读取
```
后果：多次 kernel launch 开销叠加，输入侧 GM 读流量随 kernel 数成倍增加；该 kernel 本身是 memory-bound，分离写法直接损失约 25% 收益。✅ 正解：融合为单 kernel，切分边界（q_end、k_end 等）全部定义为 `tl.constexpr`，B/A 两段循环用 `tl.static_range` 全展开，一次顺序读入、五路连续写出。

**已知实例**：misc §raw (`experiences/misc/exp_fused_qkvzba_split_001.yaml`)
- 声称性能收益：25%
- 备注：This kernel is primarily memory-bound. The performance gains are derived from reducing kernel launch latency and optimizing memory access patterns through fusion and compile-time unrolling. Ensure input tensors are contiguous in memory for maximum efficiency.

<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](../algorithm-numerics/split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: gemm based histogram](../compute-pipeline/exp_digit_hist_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: grid stride loop](../compute-pipeline/exp_element_mul_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: split k parallelism](../compute-pipeline/exp_flashattn_splitk_gqa_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: operator fusion](../compute-pipeline/exp_fused_index_001.md) — 主题相关：共享 misc
- 相关主题: [Scheduling Optimization: deep operator fusion](../compute-pipeline/exp_fused_transformer_block_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: numerical stability](../compute-pipeline/exp_grpo_loss_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 主题相关：共享 misc
- 相关主题: [计算优化: phase split 与 barrier 外提](../compute-pipeline/exp_kda_bwd_wy_structure_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: compile time dispatch](../compute-pipeline/exp_radix_convert_001.md) — 主题相关：共享 misc
- 相关主题: [Compute Optimization: scalar loop vectorization](../compute-pipeline/exp_spec_decode_verify_scalar.md) — 主题相关：共享 misc
- 相关主题: [tl.trans 标量化灾难与寄存器重排替代（split/interleave/reshape 安全集）](../compute-pipeline/triton-ascend-trans-scalarization.md) — 同属优化知识，主题相关
- 相关主题: [按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径](../meta-skills/multi-strategy-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sub blocking strategy](exp_gn_wb_backward_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: fused lse reduction](exp_grpo_bwd_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kernel fusion](exp_logits_penalty_fusion_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: redundant access elimination](exp_masked_fill_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion licm](exp_outer_product_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 访存效率
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: intermediate data residency](exp_spec_dec_reject_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized comparison](exp_spec_dec_verify_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: vectorized chunking](exp_speculative_recovery_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 misc, 访存效率
- 相关主题: [Memory Access Optimization: software pipelining](exp_tv_distance_001.md) — 主题相关：共享 misc, 访存效率

<!-- okf:related:end -->
