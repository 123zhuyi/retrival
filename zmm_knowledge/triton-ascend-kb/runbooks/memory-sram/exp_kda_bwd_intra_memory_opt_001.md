---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 内存访问优化：转置 load 行优先化与跨阶段复用
description: ssm 场景下，内存访问优化：转置 load 行优先化与跨阶段复用，收益 35%
tags: [访存效率, ssm]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存访问优化：转置 load 行优先化与跨阶段复用

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16，K≥64、T≥128，chunk 内存在转置访问的 backward kernel；失效边界：转置目标块小于向量化粒度时行优先加载+tl.trans 的指令开销抵消合并访存收益；寄存器复用跨越 tl.debug_barrier() 时可能触发 spill，需实测验证）

### 原则
- **瓶颈**：AIV MTE2=37.9%，主要由转置load非合并访存和冗余load导致
- **优化1-转置load合并化**：Phase 2中dAqk/dAkk转置加载从stride=(1, H*BT)改为行优先加载+tl.trans寄存器转置
- **优化2-消除冗余load**：Phase 1.5加载的b_q在Phase 2.5复用；Phase 1加载的b_b在Phase 2.5复用；b_g在Phase 1.5复用
- **优化3-手动指针load替代block ptr**：虽然主要目标是消除scalar，但也减少block_ptr结构体创建开销
- **注意**：寄存器复用需考虑tl.debug_barrier()是否导致spill，需实测验证
- **预期效果**：AIV MTE2从37.9%降至<25%

### 代码/骨架出处
- `Triton`:
```python
# 转置 load 合并化
# 反模式：列方向 stride=H*BT
p_dAqk = tl.make_block_ptr(dAqk_local, (BT, T_local), (1, H*BT),
                         (i_i*BC, i_t*BT+i_j*BC), (BC, BC), (0, 1))
b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))

# 正模式：行优先 + 寄存器转置
row_idx = i_t * BT + i_j * BC + o_i[:, None]
col_idx = i_i * BC + o_i[None, :]
dAqk_mask = (row_idx < T_local)[:, None] & (col_idx < BT)[None, :]
p_dAqk_row = dAqk_local + row_idx * H * BT + col_idx
b_dAqk_row = tl.load(p_dAqk_row, mask=dAqk_mask, other=0.0)
b_dAqk = tl.trans(b_dAqk_row)
```

**坏实践（反例）**：直接用 block_ptr 按列方向 stride=H*BT 做转置 load，且各 Phase 重复加载同一张量
```python
p_dA = tl.make_block_ptr(dA, (BT, T_local), (1, H*BT),
                         (i_i*BC, i_t*BT+i_j*BC), (BC, BC), (0, 1))
b_dA = tl.load(p_dA, boundary_check=(0, 1))   # 列向 stride，GM→UB 非连续
# Phase 2.5 再次 tl.load 已在 Phase 1/1.5 加载过的 b_q / b_b / b_g
```
后果：AIV MTE2 占比高达 37.9%（转置 load 非合并访存 + 冗余 load 叠加）。✅ 正解：改为行优先 load（行内地址连续）+ `tl.trans` 寄存器转置；b_q/b_b/b_g 在首次加载的 Phase 之后直接复用，AIV MTE2 从 37.9% 降至 <25%。

**已知实例**：ssm §raw (`experiences/ssm/exp_kda_bwd_intra_memory_opt_001.yaml`)
- 声称性能收益：35%
- 备注：此经验与 exp_perf_015 通用转置 load 合并化技术对应，但提供了该类分 chunk 状态递推 backward kernel 的具体上下文和 mask 计算示例。

<!-- okf:related:start -->

# 相关

- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized offset generation](exp_attn_causal_mask_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block pointer abstraction](exp_attn_ptr_setup_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax](exp_attn_sinks_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sparse gather optimization](exp_attn_sparse_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: vectorized 3d memory access](exp_cache_store_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: block ptr utilization](exp_causal_conv1d_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: sliding window register reuse](exp_causal_conv1d_prefill_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: make block ptr hardware acceleration](exp_causal_conv1d_update_003.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: chunked recurrence strategy](exp_chunk_gated_delta_001.md) — 主题相关：共享 ssm, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: 2D Tiling Strategy](exp_embedding_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: tiling strategy](exp_flash_attn_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: online softmax algorithm](exp_flashattn_fwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fuse_tlx_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_add_layernorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](exp_fused_attention_002.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: operator fusion](exp_fused_norm_rope_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](exp_fused_qkvzba_split_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused split and dual-norm single-pass load](exp_fused_split_qk_norm_001.md) — 主题相关：共享 访存效率

<!-- okf:related:end -->
