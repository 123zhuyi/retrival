---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: make block ptr hardware acceleration"
description: "conv 场景下，Memory Access Optimization: make block ptr hardware acceleration，收益 block_ptr vs scalar load: 在 D≥2048 场景差异可达 2"
tags: [访存效率, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: make block ptr hardware acceleration

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3，tile 加载为性能敏感路径；失效边界：D 较小（<2048）时 block_ptr 与标量 load 差异收窄，混合方案的 extract_slice 开销需实测确认；地址不规则的 gather 场景 block_ptr 不适用）

### 原则
- **make block ptr hardware acceleration**：`tl.make_block_ptr` 在 Ascend NPU 上映射为硬件 block transfer 指令（类似 DMA），一条指令搬运整个 tile，包含自动边界检查和零填充（padding_option='zero'）。这是 Ascend 专为 tile 加载优化的硬件原语，效率远高于通用标量 load 指令。在 causal_conv1d_update 中用于加载权重 `[D_CHK_SIZE, W]` 和存储输出 `[1, D_CHK_SIZE, T_CHK_SIZE]`。
- **scalar load degradation**：用 4 次独立 `tl.load(weight_ptr + d_offs * width + offset, mask=mask_w)` 替代 1 次 `tl.load(make_block_ptr(...))` 会导致显著性能倒退。每条独立 load 需要独立的标量地址计算，丧失 block transfer 的 DMA 级效率。这在 W_IS_4 展开优化的负收益 case 中是首要根因。
- **block ptr boundary check**：`make_block_ptr(..., boundary_check=(0, 1))` 由硬件处理边界检查，不产生软件 mask 运算开销。配合 `padding_option='zero'`，越界访问自动填零，消除 kernel 内的显式 mask 分支。
- **hybrid loading strategy**：方案 B（进阶）：保留 make_block_ptr 加载权重 tile `[D_CHK_SIZE, W]`，然后展开 W 循环手动提取各列 `w0=tl.extract_slice(w, (0,0), (D_CHK,1))`，同时消除 x_b/w/out 的 3 次 transpose。这样既保留了 block transfer 的硬件加速，又获得了循环展开的数据编排消除收益。需实测验证大 D 时 w 的 4 次 extract_slice 开销是否可忽略。
- **weight prefetch leverage**：make_block_ptr 提供硬件预取提示，允许 Ascend MTE2 在 Cube 计算期间异步预取下一 tile 的权重。独立 tl.load 无法提供等效的预取语义，导致 MTE2→Vector 流水线停顿增加。

### 代码/骨架出处
- `hybrid_approach`:
```python
# 方案 B: 保留 block_ptr 加载 + 消除 transpose
w = tl.load(make_block_ptr(weight_ptr, shape=(dim, width), ...,
          offsets=(di * D_CHK_SIZE, 0), block_shape=(D_CHK_SIZE, width)))
# w 保持 [D_CHK, 4]，无需 transpose
w0 = tl.extract_slice(w, (0, 0), (D_CHK, 1), (1, 1))
w1 = tl.extract_slice(w, (0, 1), (D_CHK, 1), (1, 1))
w2 = tl.extract_slice(w, (0, 2), (D_CHK, 1), (1, 1))
w3 = tl.extract_slice(w, (0, 3), (D_CHK, 1), (1, 1))
# x_b 保持 [D_CHK, buf] 布局，直接提取 4 个滑动窗口切片
x_s0 = tl.extract_slice(x_b, (0, 0), (D_CHK, T_CHK), (1, 1))
x_s1 = tl.extract_slice(x_b, (0, 1), (D_CHK, T_CHK), (1, 1))
x_s2 = tl.extract_slice(x_b, (0, 2), (D_CHK, T_CHK), (1, 1))
x_s3 = tl.extract_slice(x_b, (0, 3), (D_CHK, T_CHK), (1, 1))
out_block = (x_s0 * w0 + x_s1 * w1 + x_s2 * w2 + x_s3 * w3)
```

**坏实践（反例）**：为配合 W 循环展开，把权重 tile 的 block_ptr 加载退化为 W 次独立标量 load
```python
w0 = tl.load(weight_ptr + d_offs * width + 0, mask=mask_w)  # 4 次独立标量 load
w1 = tl.load(weight_ptr + d_offs * width + 1, mask=mask_w)
w2 = tl.load(weight_ptr + d_offs * width + 2, mask=mask_w)
w3 = tl.load(weight_ptr + d_offs * width + 3, mask=mask_w)
```
后果：每次 load 都要独立标量地址计算，丧失硬件 block transfer（DMA 级）效率与 MTE2 异步预取，D≥2048 时与 block_ptr 方案差 2-3x；这正是 W_IS_4 展开优化出现负收益的首要根因。✅ 正解：混合方案——保留 `make_block_ptr` 整块加载 `[D_CHK_SIZE, W]`，再用 `tl.extract_slice` 提取各列，硬件加速与数据编排消除兼得。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_update_003.yaml`)
- 声称性能收益：block_ptr vs scalar load: 在 D≥2048 场景差异可达 2-3x（与循环展开组合效应）
- 备注：此经验阐明 Ascend 上 make_block_ptr 与标量 tl.load 的根本差异：前者是硬件 DMA 原语，后者是通用标量指令。在性能敏感的 tile 加载路径上，应优先使用 make_block_ptr。若需手动索引权重（如 W 循环展开），应采用混合方案（block_ptr 加载 + extract_slice 切片）而非退化到独立 tl.load。

<!-- okf:related:start -->

# 相关

- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_block_inverse_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr utilization](../compute-pipeline/exp_block_matmul_inv_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](../compute-pipeline/exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](../compute-pipeline/exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](../compute-pipeline/exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](../compute-pipeline/exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../compute-pipeline/exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](../compute-pipeline/exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: block ptr tiling](../compute-pipeline/exp_chunk_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: block ptr abstraction](../compute-pipeline/exp_chunk_kkt_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: hardware unit isolation](../compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: hardware specific instruction](../compute-pipeline/exp_matmul_qn_001.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_paged_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr abstraction](exp_sdpa_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_topk_pad_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
