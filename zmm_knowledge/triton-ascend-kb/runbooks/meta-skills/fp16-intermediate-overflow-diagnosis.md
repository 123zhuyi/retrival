---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.2"
title: fp16 中间步骤溢出诊断--逐层定位首个溢出点
description: fp16 算子中间结果溢出时逐层打印 finite/min/max 定位首个溢出层，常见于大常数乘法/outer product/exp 大输入
tags: [fp16_overflow, layer_by_layer, finite_check, intermediate_overflow, diagnosis]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## DG-6 fp16 中间步骤溢出诊断

【优化维度】：元技能｜精度修复链
【泛化层级】：条件性（迁移条件：fp16 算子中间结果溢出（nan/inf 出现）；失效边界：纯 fp32 计算无溢出风险）

<判定依据：fp16 最大值约 65504，中间步骤的大常数乘法（如 *1000）、outer product、exp/sin/cos 大输入极易溢出。直接看最终输出 nan/inf 无法定位根因，必须逐层打印 finite count + min/max 定位首个溢出层。与 [[fp32-reduction-accumulator]] 互补：N1 是累加精度，本条是中间溢出定位。>

<诊断动作时序（逐层下钻）：
1. 建逐层 stats：每层计算后打印 shape / finite_count / total / min / max
2. 从输入逐层推进：input -> 中间层1 -> 中间层2 -> ... -> output
3. 定位首个 finite_count < total 的层（首个溢出层）
4. 检查该层的运算类型：
   - 大常数乘法（*1000.0）-> fp16 *1000 易溢出（65536/1000=65 上限）
   - outer product（unsqueeze * unsqueeze）-> 元素级放大
   - exp/sin/cos 大输入 -> exp(>11) 即溢出 fp16
   - linear/matmul 累加 -> 多元素累加溢出
5. 修复：该层升 fp32 计算，或 clamp 输入，或拆分常数
>

<骨架（占位名伪码，逐层诊断）：
def stats(t, name):
    finite = torch.isfinite(t)
    print(f"{name}: finite={finite.sum()}/{t.numel()} min={t.min()} max={t.max()}")
stats(x, "input")
x1 = x * 1000.0;          stats(x1, "after *1000")      # 常见溢出点
x2 = x1.unsqueeze(-1) * freqs.unsqueeze(0); stats(x2, "after outer")  # outer product
x3 = torch.cat([torch.sin(x2), torch.cos(x2)], -1); stats(x3, "after sin/cos")
x4 = F.linear(x3, w1, b1); stats(x4, "after linear1")
# 首个 finite 下降的层即溢出根因
>

**坏实践（反例）**：只看最终输出 nan 就猜根因
```python
out = model(x)  # nan
# 猜测：可能是权重问题 -> 改权重（错）
# 猜测：可能是 lr 问题 -> 改 lr（错）
```
后果：未定位到中间溢出层，反复试错。✅ 正解即逐层 stats 定位首个溢出点，该层升 fp32。

**已知实例**：op_0_10/op_10_FluxTimestepGuidanceProjectionEmbedding（timestep*1000 + outer product 溢出，show_flux_overflow.py 逐层诊断）

<!-- okf:related:start -->

# 相关

- 相关主题: [Memory Access Optimization: register reuse intermediate](../memory-sram/exp_rmsnorm_add_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: intermediate data residency](../memory-sram/exp_spec_dec_reject_001.md) — 同属优化知识，主题相关
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](attention-fusion-failure-diagnosis.md) — 主题相关：共享 diagnosis
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](dot-vs-matmul-precision-diagnosis.md) — 主题相关：共享 diagnosis
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](pipeline-diagram-driven-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
