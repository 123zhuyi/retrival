---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 带 alpha 缩放的逐元素 add 访存与块大小优化
description: elementwise 场景下，带 alpha 缩放的逐元素 add 访存与块大小优化，收益 1.67x（平均时延 2.50ms 降至 1.50ms）
tags: [访存效率, general, elementwise]
confidence: unverified
status: draft
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 带 alpha 缩放的逐元素 add 访存与块大小优化

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp32/fp16/bf16，支持广播的逐元素 add（x + alpha*y）；失效边界：归约/扫描等非逐元素模式不适用；多算子复合链需单独调优；块大小档位为 910b2 标定，换硬件需重测）

### 原则
- **block size tuning**：按张量维度动态选择块大小，避免小张量下向量单元空转、大张量下 program 数膨胀带来的调度开销。
- **continuous memory access**：采用 GM→UB 连续访存模式组织 load/store，配合 mask 处理尾部越界元素，保证边界正确性。
- **broadcast alpha scaling**：利用 Triton 广播能力实现 x + alpha * y，标量 alpha 直接参与向量运算，无需额外张量读写。
- 实测收益：平均时延 2.50ms → 1.50ms，相对 PyTorch baseline 加速 1.67x（910b2，fp32/fp16/bf16）。

**坏实践（反例）**：块大小一刀切且不做边界 mask
```python
BLOCK_SIZE = 1024
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offs)            # 无 mask，尾部越界
y = tl.load(y_ptr + offs)
tl.store(out_ptr + offs, x + alpha * y)
```
后果：小张量时向量单元空转、大张量时 program 数膨胀；尾部越界读写导致结果错误。✅ 正解：按张量维度分档选择块大小，所有 load/store 带 `mask=offs < n_elements` 处理尾部。

**已知实例**：elementwise §raw (`experiences/elementwise/exp_elemwise_add_001_20260526.yaml`)
- 声称性能收益：1.67x（平均时延 2.50ms → 1.50ms，vs PyTorch baseline）
- 备注：适用于带广播与 alpha 标量参数的逐元素加法场景。

<!-- okf:related:start -->

# 相关

- 实践案例: [Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡](../../ops/add-broadcast-alpha.md) — 主题相关：共享 elementwise
- 实践案例: [ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡](../../ops/apply-rotary-pos-emb.md) — 主题相关：共享 elementwise
- 实践案例: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 主题相关：共享 elementwise
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 主题相关：共享 elementwise
- 实践案例: [FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡](../../ops/fused-bias-leaky-relu.md) — 主题相关：共享 elementwise
- 实践案例: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](../../ops/moe-finalize-routing-v2-grad.md) — 主题相关：共享 elementwise
- 实践案例: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](../../ops/moe-finalize-routing-v2.md) — 主题相关：共享 elementwise
- 实践案例: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 主题相关：共享 elementwise
- 实践案例: [NsaCompressGrad (NSA compress backward) on Ascend NPU](../../ops/nsa-compress-grad.md) — 主题相关：共享 elementwise
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 elementwise
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 elementwise
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 elementwise
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 elementwise
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [Memory Access Optimization: fp8 value casting](exp_attn_fp8_ub_slice_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: flattened grid scheduling](exp_causal_conv1d_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: compile time branching](exp_causal_conv1d_states_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused state update](exp_causal_conv1d_update_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: overlapping t tiling](exp_causal_conv1d_update_004.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 elementwise, general, 访存效率
- 相关主题: [Memory Access Optimization: simd index select](exp_flashattn_gqa_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused residual-add RMSNorm with in-place writeback](exp_fused_rms_norm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fused add-RMSNorm with weight hoisting](exp_gemma_rmsnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: dynamic vectorization tuning](exp_groupnorm_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：转置 load 行优先化与跨阶段复用](exp_kda_bwd_intra_memory_opt_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: adaptive tiling strategy](exp_layernorm_bwd_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: loop invariant code motion](exp_lightning_attn_decode_001.md) — 主题相关：共享 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](exp_perf_016.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Memory Access Optimization: dynamic block sizing with fused backward+add](exp_rmsnorm_bwd_fused_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](exp_rmsnorm_fused_quant_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](exp_solve_tril_002.md) — 主题相关：共享 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率

<!-- okf:related:end -->
