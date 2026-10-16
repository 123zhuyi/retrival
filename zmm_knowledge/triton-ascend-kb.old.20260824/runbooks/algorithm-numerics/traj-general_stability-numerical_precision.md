---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_stability
dimension: "3.4"
title: 数值精度优化：fp16/bf16 下的 fp32 计算与累加
description: 聚合 2 条 general_stability 数值精度经验，核心结论：在 Ascend 910B 上，复杂计算和大步数累加必须在 fp32 上完成，否则精度会显著漂移。
tags: [general_stability, numerical_precision, fp32, fp16, bf16, accumulation, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_stability/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# 数值精度优化：fp16/bf16 下的 fp32 计算与累加

## 概述

在 LLM 训练和推理中，激活、权重、梯度常以 fp16 或 bf16 存储以节省内存和带宽。但 fp16 的动态范围有限（≈ 6×10⁻⁵ ~ 65504），bf16 虽然范围与 fp32 相同，但尾数只有 7 位，**中间计算如果直接在低精度上执行，会迅速丢失有效位**。本卡聚合 2 条数值精度经验，核心结论非常一致：

- **复杂计算步骤**（多层激活、累积求和）应在 fp32 上执行；
- **大步数累加**（梯度累积、loss 求和、统计归约）必须使用 fp32 累加器；
- 只在最终结果存储前 downcast 回 fp16/bf16。

这本质上是用一点额外的 Vector 计算换取训练稳定性和推理正确性。

## 共同模式

### 1. 核心计算步骤启用 fp32

适用场景：
- LayerNorm/RMSNorm 的 mean/variance 计算；
- Softmax 的 exp 和求和；
- Attention 的 scale + softmax + weighted sum；
- 多层激活嵌套（如 SwiGLU、GELU 后的矩阵乘）。

根因：fp16 的乘法和加法会截断小数位，连续多步后误差会累积并放大。bf16 虽然范围足够，但精度只有 7 bit，对需要精细权重的场景（如 attention score）同样不够。

标准做法：
```python
x = tl.load(x_ptr + offsets).to(tl.float32)  # 加载后立即 upcast
# ... 复杂计算 ...
tl.store(out_ptr + offsets, result.to(out_dtype))  # 存储前 downcast
```

### 2. 累加操作使用高精度累加器

适用场景：
- 梯度累积（gradient accumulation）；
- Loss 跨 batch/token 求和；
- LayerNorm/RMSNorm backward 的 dW 累加；
- 长序列 attention 中的 online softmax running sum。

根因：fp16 累加器在 1000 次累加后误差可达 ~1e-3，而 fp32 累加器误差仅 ~1e-7。对于 LLM 这种动辄数万 token、多层网络的场景，fp16 累加会导致模型不收敛或 loss 曲线异常。

标准做法：
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(...):
    acc = tl.dot(a, b, acc=acc, allow_tf32=False)  # GEMM 必须用 fp32 累加
```

### 3. 性能与精度的权衡

把中间计算提升到 fp32 会带来：
- 更多的 Vector 单元计算量（但通常不是瓶颈）；
- 稍高的寄存器占用（可通过 tiling 控制）；
- 无额外 HBM 带宽（因为输入/输出仍是低精度）。

实际经验：在归约、softmax、norm、loss 等带宽受限型算子中，fp32 中间计算的额外开销通常小于 5%，但能把 MAE 降低 80%、MSE 降低 90%。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_stability_003` | MAE 降低 80% | 多层激活/累积求和场景，核心计算步骤启用 fp32 |
| `exp_stability_004` | MSE 降低 90% | 大步数累加/梯度累积/loss 求和场景，使用 fp32 累加器 |

## 代表 Kernel

以下为体现本卡核心原则的示意代码（以 LayerNorm 为例）：

```python
@triton.jit
def rmsnorm_fp32_kernel(
    x_ptr, w_ptr, y_ptr,
    n_rows, n_cols, eps: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_COLS)
    mask = col_offsets < n_cols

    # 1. 加载后立即 upcast 到 fp32
    x = tl.load(x_ptr + row_idx * n_cols + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)

    # 2. 所有核心计算在 fp32 完成
    x_sq = x * x
    variance = tl.sum(x_sq, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)
    y = x * rstd * w

    # 3. 存储前 downcast
    tl.store(y_ptr + row_idx * n_cols + col_offsets,
             y.to(y_ptr.dtype.element_ty), mask=mask)
```

关键实现细节：
- `x` 和 `w` 加载后立即 `.to(tl.float32)`；
- 求和、除法、rsqrt、乘法全在 fp32；
- 只在 `tl.store` 前 downcast 到输出 dtype。

## 已知实例

- `exp_stability_003` — Numerical Precision（收益：MAE 降低 80%）
  - 适用于多层激活、累积求和等复杂计算场景。fp16/bf16 动态范围有限，中间计算易丢失小数位，910B 平台建议所有核心计算步骤启用 float32 转换。
- `exp_stability_004` — Numerical Precision（收益：MSE 降低 90%）
  - 适用于大步数累加、梯度累积、损失求和等场景。fp16 累加器在 1000 次累加后误差可达 1e-3，float32 累加器误差仅 1e-7，910B 平台建议所有累加操作启用高精度累加器。

## 关联知识

- 开发指南: [Normalization 算子优化](../../reference/op-class-normalization.md)
- 相关 runbook: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](../algorithm-numerics/fp32-reduction-accumulator.md)
- 相关 runbook: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md)
- 相关 runbook: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [Elementwise absolute value (Abs)](../../ops/elementwise-abs.md) — 主题相关：共享 bf16
- 相关主题: [Elementwise add with a tail mask](../../ops/elementwise-add.md) — 主题相关：共享 bf16
- 相关主题: [MoeFinalizeRouting (MoE FFN output routing merge) on Ascend NPU](../../ops/moe-finalize-routing.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [MoeTokenPermuteWithEp (MoE token permute with Expert Parallel) on Ascend NPU](../../ops/moe-token-permute-with-ep.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](../../ops/moe-token-unpermute.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 主题相关：共享 bf16, fp16
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 bf16
- 相关主题: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 bf16, fp16, fp32
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 bf16
- 相关主题: [Activation Scheduling 优化聚合](../compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, fp32
- 相关主题: [bf16 融合反量化写单表达式解锁 BLOCK_K=128](../memory-sram/bf16-fused-dequant-blockk128.md) — 主题相关：共享 bf16
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](../meta-skills/fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [反量化域必须按 x.dtype 分流（fp16 域 vs fp32 域）](dequant-domain-dtype-split.md) — 同属优化知识，主题相关
- 相关主题: [统计归约必须用 fp32 累加器，禁止 fp16/bf16 直接累加](fp32-reduction-accumulator.md) — 同属优化知识，主题相关
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3, bf16, fp16, general_stability

<!-- okf:related:end -->
