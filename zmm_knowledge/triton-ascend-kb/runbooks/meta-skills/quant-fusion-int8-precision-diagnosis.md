---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.2"
title: 量化融合算子 int8 输出精度不一致的逐层下钻诊断链
description: 量化融合算子（matmul+activation+quantize）int8 output 与参考不一致时，按 output->scale->hidden->activation 逐层下钻定位根因的决策链
tags: [precision_debug, quantization, int8, sigmoid, expert_boundary, diagnostic_chain, step_up]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## DG-1 量化融合算子 int8 输出精度不一致诊断链

【优化维度】：元技能｜精度修复链
【泛化层级】：Triton通用（迁移条件：量化融合算子 matmul+activation+quantize 链路 int8 output 与参考不一致；失效边界：纯 fp32 算子精度问题不走此链，直接查累加顺序/dtype 转换）

<判定依据：量化融合算子精度失败表现为 int8 output diff，但根因可能在反量化、matmul 累加、激活函数实现、或 expert 边界映射。直接看 int8 output 无法定位根因，必须逐层下钻到中间 fp32 张量。与 [[fp32-reduction-accumulator]] 配套：下钻到累加层时套用 fp32 累加原则。>

<诊断决策树（从 output 向上游下钻，定位首个大发散层）：

1. output(int8) diff > 0
   ├─ 检查 output_scale diff
   │  ├─ scale diff 大 -> 量化阶段 abs_max 计算错误（检查 per-token vs per-group scale 路径分发）
   │  └─ scale diff 小、output diff 大 -> 量化前 fp32 值已不一致 -> 下钻
   ├─ 下钻到 quantize 前 fp32 张量（y_fp32 / activation_out）diff
   │  ├─ y diff 大 -> 激活阶段问题 -> 步骤 3
   │  └─ y diff 小 -> 量化 round/clamp 实现问题
   └─ 下钻到 hidden（matmul 输出）diff
      ├─ hidden diff 大 -> matmul/dequant 阶段 -> 步骤 2
      └─ hidden diff 小、y diff 大 -> 激活函数实现差异 -> 步骤 3

2. matmul/dequant 阶段定位（hidden diff 大）
   ├─ 检查 x_scale/weight_scale 反量化（int8->fp32）是否先升精度
   ├─ 检查 int8 dot 累加是否用 fp32 累加器（禁止 int8 直接累加）
   ├─ 单独构造最小用例隔离 matmul（如 test_int8_dot 系列，逐版本对比累加实现）
   └─ 检查 BLOCK_K 切分下累加顺序差异

3. 激活函数实现差异定位（hidden diff 小、y diff 大）
   ├─ ref 用 torch.sigmoid，triton 手写 1/(1+exp(-x))（tl.sigmoid 被禁用时）
   │  -> 检查手写 sigmoid 在大负数/大正数区的精度损失
   ├─ 检查 swish: x * sigmoid(scale*x) 的计算顺序与 dtype
   └─ 检查激活前后 dtype 转换是否丢精度

4. diff 分布特征分析（定位系统性问题，非随机噪声）
   ├─ diff 集中在 expert/group 边界行 -> expert_id_per_row 映射错误（查 _compute_expert_start/_fill_expert_id）
   ├─ diff 按 program 规律分布（grid=N, program=row%N）-> grid 切分/跨核负载问题
   └─ diff 随机分布 -> 纯数值精度（累加顺序、dtype 转换）

5. 特定位置深挖（max diff 位置 [r,c]）
   ├─ 打印 ref/triton 在该位置的 hidden/activation/scale 逐值
   ├─ 检查 weight_scale 符号（负值可能引发 sign flip）
   └─ 检查 expert_id 是否正确（影响取哪组 weight）
>

<动作时序（推荐诊断顺序，颠倒则无效迭代）：
1. 建双路 Debug：Ref 与 Triton 实现都返回中间张量（hidden/activation/scale），不加中间量无法下钻
2. 逐层比 diff：output -> scale -> y_fp32 -> hidden -> activation，从后往前定位首个大发散层
3. 首个大发散层定位后，构造该层最小用例隔离（如 test_int8_dot 隔离 matmul 精度）
4. 分析 diff 分布特征（边界行 / program 规律 / 随机）排除系统性问题
5. 取 max diff 位置深挖具体数值，定位根因后修复
>

**坏实践（反例顺序）**：直接调 int8 output 的 round/clamp 阈值或 scale 偏移试图对齐
- 后果：掩盖上游精度问题，换 shape 又失败；本轨迹 iter_0-19 共 20 轮，部分轮次因未下钻到根因而无效迭代
- ✅ 正解：先逐层下钻定位首个大发散层，再针对性修复根因

**已知实例**：GroupedMatmulSwigluQuantV2 case26（`triton-op-generator/triton_ascend_output/op_0_11_GroupedMatmulSwigluQuantV2_20260706_1234_6600/debug_case26.py`，20 轮迭代，根因为手写 sigmoid 精度 + expert 边界映射）

<!-- okf:related:start -->

# 相关

- 实践案例: [KvQuantSparseFlashAttention (sparse inference with KV quant) on Ascend NPU](../../ops/kv-quant-sparse-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 同属优化知识，主题相关
- 实践案例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingQuantV2 (MoE routing init V2 + dynamic quant + drop) on Ascend NPU](../../ops/moe-init-routing-quant-v2.md) — 主题相关：共享 int8
- 实践案例: [MoeInitRoutingQuant (MoE routing init + quantize) on Ascend NPU](../../ops/moe-init-routing-quant.md) — 主题相关：共享 int8
- 实践案例: [MoeInitRoutingV3 (MoE routing with dynamic/static quant) on Ascend NPU](../../ops/moe-init-routing-v3.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermute (MoE token permute by indices) on Ascend NPU](../../ops/moe-token-permute.md) — 主题相关：共享 int8
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [QuantLightningIndexer (SFA quantized sparse token selection) on Ascend NPU](../../ops/quant-lightning-indexer.md) — 主题相关：共享 int8
- 相关主题: [fp32 精度契约（长链复合算子低精度 verify 不可达的破解模式）](../algorithm-numerics/fp32-precision-contract.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_attn_bwd_q_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: mixed precision compute](../compute-pipeline/exp_flashattn_bwd_kv_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: high precision accumulation](../compute-pipeline/exp_recompute_w_u_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: mixed precision computation](../compute-pipeline/exp_swiglu_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 quantization
- 相关主题: [Memory Access Optimization: fusing add/norm/bias/quant into one kernel](../memory-sram/exp_rmsnorm_fused_quant_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: precision control](../memory-sram/exp_skip_layernorm_001.md) — 同属优化知识，主题相关
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [fp16 中间步骤溢出诊断--逐层定位首个溢出点](fp16-intermediate-overflow-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [流水图驱动优化：先可视化找气泡，按气泡边类型分派优化手段](pipeline-diagram-driven-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
