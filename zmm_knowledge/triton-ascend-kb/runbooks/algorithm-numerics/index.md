---
kind: index
type: section_index
title: runbooks/algorithm-numerics/（占位）
description: 分类3 算法与数值优化点库（OPT-*/AP-*/CT-*，跨算子单一共享、增量合并）。
---

# 分类3 算法与数值（占位）

覆盖二级维度：3.1 算法重构 · 3.2 混合精度 · 3.3 数值稳定 · 3.4 跨 Block 归约。

## 新增条目（P2 第1批 + 剩余）

- [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — CT-12，fp16 exp 溢出防护与 fp32 累加
- [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — OPT-24，O(S) 内存的 online softmax


## 新增条目（raw experiences 导入，待 review）

- [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](exp_stability_001.md) — 910b2/910b3 + fp16/bf16 下指数运算前先减维度内最大值，消除 exp(大数) 返回 Inf/NaN；代价为一次 max 归约（硬件 max 指令耗时可忽略），适用于 softmax、注意力分数等含 exp 场景
- [sqrt 前用 max(x, eps) 钳位防负数开方 NaN（eps 按精度选型）](exp_stability_002.md) — 910b2/910b3 上硬件 sqrt 对负数输入直接返回 NaN，开方前用 tl.maximum(x, eps) 钳位——fp16/bf16 建议 eps=1e-6、fp32 建议 1e-8，解决数值误差产生的微小负数开方问题
- [关键计算步骤转 fp32 执行再转回原精度（MAE 降 80%，算力代价约 5%）](exp_stability_003.md) — 910b2/910b3 上多层激活/累积求和等复杂计算，中间步骤转 float32 执行、完成后转回 fp16/bf16 存储；声称 MAE 降 80%+，fp32 计算耗时仅高 5%，精度收益远大于性能损耗
- [累加器必须 fp32 初始化：fp16 累加 ≥1000 次漂移明显（MSE 降 90%）](exp_stability_004.md) — 910b2/910b3 上大步数累加/梯度累积/损失求和场景，累加器用 tl.zeros(..., dtype=tl.float32) 初始化；fp16 累加器 1000 次累加误差达 1e-3，fp32 仅 1e-7，声称 MSE 降 90%

> 其余条目由摄入流程增量合并产生。

## trajectory 实例（P3/P5 提取）

- [attention Memory Access 实践聚合](traj-attention-memory_access.md) — 22 条 attention 访存优化 experience 聚合
- [attention Compute 实践聚合](traj-attention-compute.md) — 11 条 attention 计算优化 experience 聚合
- [attention Memory Numerical 实践聚合](traj-attention-memory_numerical.md) — 1 条 attention 内存+数值联合优化 experience 聚合
- [normalization Memory Access 实践聚合](traj-normalization-memory_access.md) — 24 条 normalization 访存优化 experience 聚合
- [normalization Compute 实践聚合](traj-normalization-compute.md) — 7 条 normalization 计算优化 experience 聚合
- [general_stability Numerical Stability 实践聚合](traj-general_stability-numerical_stability.md) — 2 条数值稳定性 experience 聚合
- [general_stability Numerical Precision 实践聚合](traj-general_stability-numerical_precision.md) — 2 条数值精度 experience 聚合


## 新增条目/实例（2026-09-21）

- [online-softmax-flash-attention](online-softmax-flash-attention.md) — append 已知实例：SelfAttention（mha, ascend910b）FA 64x128 KV 迭代减半 +49%；MIX_AIC 天花板 ~40x 不可闭合
  - （2026-09-30）再 append：SelfAttention 互补 merge（FULL_D 满块分裂 + tile 阶梯 + fp32 表）0.5577→0.7151（+28.2%），超市前逐 case 理想互补预算——FULL_D 分裂惠及全体主流 case
## 新增条目（staging 合入，2026-08-27）

- [fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）](fp32-precision-contract.md) — ≥10-op 非线性链输入全升 fp32、出口降回标称 dtype
