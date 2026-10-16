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
updated_at: 2026-09-22T11:44:30Z
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
- CrossformerAttention（分组窗口 MHA，flash_attention，ascend910b，2026-09-21）：含 softmax/exp/归一化的单 kernel attention 经 IR 确认编译为 **MIX_AIC**（`mix_mode="mix"`，并行 mode=simd，SyncBlockLockArgIdx=0），带 ~4.5us MIX 发射固定地板 + Cube↔Vector 跨核同步；此结构性下限（FA §3.7）无法经代码级优化消除，是 triton 侧 vs CANN 多算子编排（无 MIX 地板但有 S×S 中间物化）差距的主因。语义佐证本卡"softmax 需向量侧 → 融合单 kernel 天然 MIX"。证据：`op_15_CrossformerAttention_20260921_1730_3959/ir_output/_crossformer_attn_kernel_ttadapter.mlir`、`output/opt_iter_6/log.md`
- SDPA-backward（fused flash_attention_score_grad，ascend910b，2026-09-24）：fp16/bf16 dtype 分派（原生 cube）呈现**进程级非确定性 NaN**——全量 verify 50 case 复现 NaN（24/50 过，次次一致），但隔离单 case（同 seed42 输入）单跑全部正常。根因未定位（排除溢出/坏卡），可疑点在进程内多 case 运行时 buffer 复用 / Triton kernel cache 上下文依赖。→ 转置布局+dS2 GM 清洗主链下，原生低精度 cube **不可交付确定性结果**，净收益（仅 1.6×，非理论 16×）不足以覆盖确定性风险，回退 fp32-ieee。证据：`op_18_ScaledDotProductAttentionBackward_20260921_1522_8458/output/opt_iter_5/verify/verify_result_optimized.json`（passed 24/total 50）

<!-- okf:related:start -->

# 相关

- 实践案例: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 主题相关：共享 attention
- 实践案例: [FusedInferAttentionScore inference (prefill+decode) on Ascend NPU](../../ops/fused-infer-attention-score.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [IncreFlashAttention inference-decode on Ascend NPU](../../ops/incre-flash-attention.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 主题相关：共享 attention, flash_attention
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 主题相关：共享 attention
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 主题相关：共享 attention
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 主题相关：共享 attention, flash_attention
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 主题相关：共享 attention, flash_attention
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 同属优化知识，主题相关
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）](../algorithm-numerics/exp_stability_001.md) — 同属优化知识，主题相关
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](../algorithm-numerics/online-softmax-flash-attention.md) — 主题相关：共享 flash_attention
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 主题相关：共享 attention
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_crossentropy_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax](../compute-pipeline/exp_sdpa_infer_ascend_001.md) — 主题相关：共享 attention
- 相关主题: [device-side gather 消除 strided tile load（访存结构重组，本例 89x）](../memory-sram/device-side-gather-strided-load.md) — 同属优化知识，主题相关
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](../memory-sram/pertensor-scale-materialize.md) — 主题相关：共享 ub_overflow
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](dot-vs-matmul-precision-diagnosis.md) — 主题相关：共享 diagnosis
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](fp16-intermediate-overflow-diagnosis.md) — 主题相关：共享 diagnosis
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](pipeline-diagram-driven-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
