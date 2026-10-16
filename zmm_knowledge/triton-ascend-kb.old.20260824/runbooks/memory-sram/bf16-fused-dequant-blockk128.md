---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.2"
title: bf16 融合反量化写单表达式解锁 BLOCK_K=128
description: bf16 反量化写成单表达式让 fp32 中间量留寄存器不物化 UB，否则 BLOCK_K=128 会 UB overflow
tags: [bf16, fused_dequant, single_expr, blockk128, register_spill]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-9 bf16 融合反量化单表达式

【优化维度】：内存｜Tiling·SRAM 预算
【泛化层级】：条件性（迁移条件：bf16 反量化+matmul 融合且想用 BLOCK_K=128；失效边界：fp16 路径或 precompute 路径）

<原则：bf16 反量化写成单表达式 ((w_i8.to(f32)+off_f32)*scale_f32).to(bf16)，让编译器把 fp32 中间量留寄存器不物化到 UB。否则 bf16 BLOCK_K=128 会 UB overflow（fp32 中间量 [BK,BN] 过大）。bf16 BLOCK_K 64->128 对齐 fp16，加速比 0.48->0.65。>

<骨架（占位名伪码）：
# 单表达式融合（fp32 中间量留寄存器）
w_dq = ((w_i8.to(tl.float32) + off_f32) * scale_f32).to(tl.bfloat16)
acc += tl.dot(x_bf16, w_dq)   # BLOCK_K=128 不 overflow
# 禁止：分步物化
# tmp1 = w_i8.to(f32)          # 物化 [BK,BN] fp32
# tmp2 = tmp1 + off_f32        # 又一个 [BK,BN]
# w_dq = (tmp2 * scale).to(bf16)  # UB overflow
>

**坏实践（反例）**：分步物化 fp32 中间量
```python
tmp = w_i8.to(tl.float32)        # [BK,BN] fp32 物化到 UB
w_dq = (tmp + off).to(bf16)
```
后果：bf16 BLOCK_K=128 时 UB overflow。✅ 正解即单表达式融合。

**已知实例**：weight-quant-matmul W7（bf16 加速比 0.48->0.65，opt_iter_3->4）

<!-- okf:related:start -->

# 相关

- 实践案例: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 主题相关：共享 bf16
- 实践案例: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 主题相关：共享 bf16
- 实践案例: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../../ops/fused-floyd-attention-grad.md) — 同属优化知识，主题相关
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 同属优化知识，主题相关
- 实践案例: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 主题相关：共享 bf16
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 主题相关：共享 bf16
- 实践案例: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 主题相关：共享 bf16
- 实践案例: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 主题相关：共享 bf16
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 主题相关：共享 bf16
- 实践案例: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 主题相关：共享 bf16
- 实践案例: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 主题相关：共享 bf16
- 实践案例: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](../../ops/moe-token-unpermute.md) — 主题相关：共享 bf16
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 bf16
- 实践案例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 bf16
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 bf16
- 实践案例: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 bf16
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 同属优化知识，主题相关
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](../algorithm-numerics/dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](../algorithm-numerics/traj-general_stability-numerical_precision.md) — 主题相关：共享 bf16
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 bf16
- 相关主题: [pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)](pergroup-blockk-align.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
