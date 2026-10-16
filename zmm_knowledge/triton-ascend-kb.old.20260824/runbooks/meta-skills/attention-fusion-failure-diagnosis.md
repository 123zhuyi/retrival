---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.1"
title: attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下
description: attention 尝试单 kernel 融合 QK+Softmax+PV 时的失败分支诊断（无workspace精度失败/有workspace性能差）
tags: [attention, fusion_failure, softmax_workspace, ub_overflow, flash_attention, diagnosis]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## DG-4 attention 单 kernel 融合失败诊断

【优化维度】：元技能｜瓶颈诊断
【泛化层级】：条件性（迁移条件：attention 算子尝试单 kernel 融合 QK+Softmax+PV；失效边界：非 attention 或已确认三 kernel 架构）

<判定依据：attention 的 softmax 中间状态（max/sum）需要跨 tile 聚合，单 kernel 融合要么省 workspace（精度失败）要么加 workspace（性能差）。大 D（如 1024）时 UB 放不下完整 q/k/v tile，加剧矛盾。与 [[two-pass-reduce-then-apply]] 同源：softmax 归约必须两阶段。>

<诊断决策树（单 kernel 融合失败分支）：
尝试单 kernel 融合 QK+Softmax+PV
├─ 方案A：无 workspace FlashAttention（D 维分块在线 softmax）
│  └─ softmax 基于不完整分块数据 -> 精度失败
│     根因：D 维分块时 softmax 的 max/sum 跨块无法在线聚合正确
├─ 方案B：按行单 kernel + workspace
│  ├─ 必须 BLOCK_M=8（UB 放不下大 tile）-> matmul 效率骤降
│  └─ speedup 0.043（远低于三 kernel 的 0.197）
│     根因：大 D（1024）UB 放不下完整 q/k/v tile，被迫小 BLOCK_M
└─ 正解：维持三 kernel 分离（QK / Softmax / PV）+ BLOCK_M 增大到 256
   - 三 kernel 让 softmax 独立用 fp32 + workspace，matmul 用大 BLOCK_M
   - 与 [[split-kernel-gm-roundtrip]] 一致：必要时拆 kernel 强制正确中间状态
>

<瓶颈定位指标：
- 大 shape（如 [1,384,1536]）speedup 0.017，小 shape（如 [1,256,16,32]）1.12 -> 典型 cube 利用率问题，大 shape BLOCK_M 不够
- 单 kernel speedup 0.043 vs 三 kernel 0.197 -> 融合未必更快，matmul 效率主导
>

**坏实践（反例）**：盲目追求单 kernel 融合
```python
# 假设单 kernel 一定更快，强行融合 QK+Softmax+PV
# 用小 BLOCK_M=8 塞进 UB -> matmul 效率骤降
```
后果：speedup 0.043，比三 kernel 差 4.6 倍。✅ 正解即三 kernel 分离 + 大 BLOCK_M。

**已知实例**：op_0_12_FusedInferAttentionScore（5 轮优化，0.0807->0.1974，方案A/B 均失败，最终三 kernel + BLOCK_M=256）

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 attention
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 attention
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 attention
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 attention
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention, flash_attention
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 attention, flash_attention
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 开发指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 attention
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 同属优化知识，主题相关
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 attention
- 相关主题: [大序列 Attention 的分块计算与 Online Softmax 数值稳定](../algorithm-numerics/traj-attention-memory_numerical.md) — 主题相关：共享 attention, flash_attention
- 相关主题: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../algorithm-numerics/traj-general_stability-numerical_stability.md) — 同属优化知识，主题相关
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 同属优化知识，主题相关
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](../compute-pipeline/async-task-cube-vector.md) — 同属优化知识，主题相关
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 flash_attention
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 主题相关：共享 ub_overflow
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](dot-vs-matmul-precision-diagnosis.md) — 主题相关：共享 diagnosis
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](fp16-intermediate-overflow-diagnosis.md) — 主题相关：共享 diagnosis
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [Attention 工作流优化：中小序列标准 Attention 的三步计算、数值稳定与维度对齐](traj-attention-workflow_opt.md) — 主题相关：共享 attention

<!-- okf:related:end -->
