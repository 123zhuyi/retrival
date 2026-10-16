---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.2"
title: 反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）
description: fp16 输入在 fp16 域反量化，bf16 输入在 fp32 域反量化后截断，对齐 cube L0C 行为
tags: [dequant, dtype_domain, fp16_domain, bf16_fp32, l0c]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-6 反量化域按 dtype 分流

【优化维度】：数值｜混合精度
【泛化层级】：条件性（迁移条件：权重反量化+matmul 融合；失效边界：纯 fp32 无须分流）

<原则：对齐 torch_npu op 的 cube L0C 行为--fp16 路径在 fp16 域反量化（精度足够且快），bf16 路径在 fp32 域反量化后截断（避免加法尾数损失再截断）。域不匹配导致 MERE 超阈值。与 [[fp32-reduction-accumulator]] 互补：本条是反量化域选择，N1 是累加精度。>

<骨架（占位名伪码）：
if x.dtype == fp16:
    w_dq = (w.to(f16) + off.to(f16)) * scale.to(f16)        # fp16 域
else:  # bf16
    w_dq = ((w.to(f32) + off.to(f32)) * scale.to(f32)).to(bf16)  # fp32 域后截断
>

**坏实践（反例）**：bf16 用 fp16 域或 fp16 用 fp32 域
```python
w_dq = (w.to(f16) + off.to(f16)) * scale.to(f16)  # bf16 输入走 fp16 域，加法尾数损失
```
后果：MERE 超阈值（fp16 9.77e-4 / bf16 7.81e-3），verify 失败。✅ 正解即按 dtype 分流。

**已知实例**：weight-quant-matmul W1（48/48 verify 验证）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](../../ops/moe-token-unpermute.md) — 同属优化知识，主题相关
- 实践案例: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 同属优化知识，主题相关
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 同属优化知识，主题相关
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../compute-pipeline/traj-matmul-split.md) — 同属优化知识，主题相关
- 相关主题: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../memory-sram/bf16-fused-dequant-blockk128.md) — 同属优化知识，主题相关
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](fp32-reduction-accumulator.md) — 同属优化知识，主题相关
- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [编译器融合消除中间 dtype 舍入时，拆分独立 kernel 强制 GM round-trip](split-kernel-gm-roundtrip.md) — 同属优化知识，主题相关
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](traj-attention-compute.md) — 同属优化知识，主题相关
- 相关主题: [数值精度优化：fp16/bf16 下的 fp32 计算与累加](traj-general_stability-numerical_precision.md) — 同属优化知识，主题相关
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](traj-general_stability-numerical_stability.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
