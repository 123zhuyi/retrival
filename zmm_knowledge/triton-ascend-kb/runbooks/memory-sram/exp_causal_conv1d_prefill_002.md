---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: "Memory Access Optimization: sliding window register reuse"
description: "conv 场景下，Memory Access Optimization: sliding window register reuse，收益 消除 75% 的 token load，实测贡献 Prefill kernel 整体加速比中的最大份"
tags: [访存效率, convolution-operations, conv]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## Memory Access Optimization: sliding window register reuse

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2/910b3/91093/91095，fp16/bf16/fp32，width=4 编译期已知、dim ≥512、seqlen ≥32；失效边界：width 可变或 >8 时命名寄存器方案无法扩展，需退回到数组+循环；seqlen 很小时状态初始化 load 占比升高，复用收益被稀释）

### 原则
- **sliding window register reuse**：将卷积窗口 `[col0, col1, col2, col3]` 维护在 4 个命名寄存器中，每次迭代只 load 一个新 token（col3），其余 3 个值从前次迭代的寄存器移位获得（col0←col1←col2←col3）。相比 reference 实现每 token 需要 4 次 load，load 次数降低 75%。对于 seq_len=2048 的典型 Prefill 场景，每个 program 节省 3×2048=6144 次全局内存 load。这是整个 kernel 中最大的单项性能提升来源。
- **register shift zero cost**：寄存器移位 `col0 = col1; col1 = col2; col2 = col3` 在编译器层面通过寄存器重命名实现，零周期开销。不同于内存拷贝或显式 shuffle，移位不产生任何实际指令。
- **initialization from state**：首次迭代从 conv_states 加载 3 个历史值到 col0/col1/col2，后续 seq_len-1 次迭代每步仅 1 次 load。窗口末尾的 3 个值恰好就是需要写回 conv_states 的最后 3 个 token，无需额外 load 即可完成状态更新——状态更新与卷积计算的 load 完全复用。
- **explicit register naming**：使用 4 个显式命名的寄存器（col0/col1/col2/col3）而非数组或 loop，使编译器能够：(1) 精确追踪每个寄存器的生命周期；(2) 生成最优的寄存器分配方案；(3) 将 `col0*w0 + col1*w1 + col2*w2 + col3*w3` 识别为可并行的 4 路 FMA 归约。
- **fma friendly expression**：单语句 4 路归约 `col0*w0 + col1*w1 + col2*w2 + col3*w3` 中 4 个乘法相互独立，编译器可发射 4 条并行向量乘法指令；加法树结构直接映射到硬件 FMA 单元（融合乘加），一条指令完成 mul+add。相比 `for w in range(4): acc += col[w]*weight[w]` 的循环累积方式，避免了跨迭代的 out_block 数据依赖。
- **width hardcoded tradeoff**：寄存器复用方案硬编码 W=4（4 个命名寄存器 + 4 个权重标量），无法支持可变 width。这是性能（零数据编排开销）与灵活性（通用 W 支持）的取舍。实践中该类序列建模架构的 width=4 是最常见配置，此取舍合理。

### 代码/骨架出处
- `register_reuse_pattern`:
```python
# 初始化：从 conv_states 加载历史 + 第一个新 token
col0 = tl.load(conv_states_ptr + 0*stride, mask=mask_d)
col1 = tl.load(conv_states_ptr + 1*stride, mask=mask_d)
col2 = tl.load(conv_states_ptr + 2*stride, mask=mask_d)
col3 = tl.load(x_ptrs)                    # 第 1 个新 token

for i in range(seq_len):
    # 4 路 FMA：单语句归约
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3
    if HAS_BIAS:
        acc = acc + bias
    if ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))     # fp32 SiLU
    tl.store(y_ptrs + i * stride_y, acc, mask=mask_d)

    # 窗口滑动：寄存器移位（零开销）
    col0 = col1
    col1 = col2
    col2 = col3
    # 加载下一个新 token
    col3 = tl.load(x_ptrs + (i+1) * stride_x, mask=mask_d)
```

**坏实践（反例）**：每个 token 都从内存重新加载整个窗口（reference 实现）
```python
for i in range(seq_len):
    acc = 0
    for w in range(4):
        acc += tl.load(x_ptrs + (i - 3 + w) * stride_x) * weight[w]  # 每 token 4 次 load
```
后果：seq_len=2048 时每个 program 多做 3×2048=6144 次全局内存 load，且 `acc +=` 循环累积引入跨迭代数据依赖，阻碍 FMA 并行。✅ 正解：窗口值驻留 4 个命名寄存器，每步移位复用、只 load 1 个新 token，单语句 4 路归约喂给 FMA。

**已知实例**：conv §raw (`experiences/conv/exp_causal_conv1d_prefill_002.yaml`)
- 声称性能收益：消除 75% 的 token load，实测贡献 Prefill kernel 整体加速比中的最大份额（与 2D Grid + DIM_IS_MULTIPLE 组合达 1.60x 平均加速比，最高 1.84x）
- 备注：滑动窗口寄存器复用是 Prefill kernel 最核心的创新，利用了 W=4 的小卷积核特性。此模式可推广到任何固定小窗口的序列处理 kernel（如移动平均、指数平滑、small-kernel 1D conv），前提是窗口大小在编译期已知且足够小（≤8）。

<!-- okf:related:start -->

# 相关

- 相关主题: [Scheduling Optimization: two dimensional grid strategy](../compute-pipeline/exp_causal_conv1d_prefill_003.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: pointer base precompute](../compute-pipeline/exp_causal_conv1d_prefill_004.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: pad slot early exit](../compute-pipeline/exp_causal_conv1d_prefill_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: conv bias silu triple fusion](../compute-pipeline/exp_causal_conv1d_prefill_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elision](../compute-pipeline/exp_causal_conv1d_prefill_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: w loop unrolling with threshold](../compute-pipeline/exp_causal_conv1d_update_002.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: dim is multiple mask elimination](../compute-pipeline/exp_causal_conv1d_update_005.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../compute-pipeline/exp_causal_conv1d_update_006.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Scheduling Optimization: bdt layout rationale](../compute-pipeline/exp_causal_conv1d_update_007.md) — 主题相关：共享 conv, convolution-operations
- 相关主题: [Compute Optimization: register accumulation](../compute-pipeline/exp_hist_count_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: cos sin reuse](../compute-pipeline/exp_rope_001.md) — 同属优化知识，主题相关
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 conv, convolution-operations, 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 conv, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: Tiled Two-Pass Reduction](exp_gn_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: paged kv cache](exp_mla_paged_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: register buffering](exp_moe_binned_scatter_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: register reuse intermediate](exp_rmsnorm_add_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: no-tiling register reuse with conflict-free dW accumulation](exp_rmsnorm_backward_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: single-pass tiling via UB residency](exp_rmsnorm_grad_dx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: grid-stride scheduling with two-pass reduction](exp_rmsnorm_infer_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: cos sin reuse](exp_rope_npu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: score-tile reuse with online softmax](exp_sdpa_flash_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
