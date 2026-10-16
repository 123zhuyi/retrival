---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: 3.3
title: exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）
description: 910b2/910b3 + fp16/bf16 下指数运算前先减维度内最大值，消除 exp(大数) 返回 Inf/NaN；代价为一次 max 归约（硬件 max 指令耗时可忽略），适用于 softmax、注意力分数等含 exp 场景
tags: [数值稳定, aiv, general_stability]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## exp 前减维度最大值防溢出（safe-exp/softmax 数值稳定）

【优化维度】：数值稳定
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：减最大值是恒等变换（softmax 类归一化下数学等价），但单独求 exp 不归一化的场景会改变数值语义，仅在后续有归一化/对数抵消时使用；fp32 场景 exp 溢出阈值高（约 88.7），输入有界时可省略；精度收益与性能代价：收益=消除 Inf/NaN（fp16 输入 >65504 或中间值 >~11 即溢出），代价=一次 max 归约（可忽略）；MERE/MARE/ULP 误差指标未测定）

### 原则
- **核心策略**：指数运算前先减去维度内最大值，消除大数指数导致的数值溢出
- **适配场景**：专门针对fp16/bf16低精度场景，该处理可100%避免exp函数返回Inf/NaN
- **无性能损耗**：仅增加一次max计算，910b2/910b3平台的硬件max指令耗时可忽略

### 代码/骨架出处
- `Triton`:
```python
@triton.jit def exp_overflow_opt_kernel(input_ptr, output_ptr, n_elements, dim: tl.constexpr=0, blockSize: tl.constexpr=128):
    pid = triton.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
    
    # 核心优化: 减去维度内最大值防止溢出
    max_val = tl.max(data, axis=dim)  # 按指定维度取最大值
    stable_data = data - max_val      # 数值稳定化
    
    # 指数计算（无溢出风险）
    exp_data = tl.exp(stable_data)
    
    # 存储结果（转回原精度）
    tl.store(output_ptr + offsets, exp_data.to(tl.float16), mask=mask)

# Softmax场景示例调用
def softmax_opt(input_tensor):
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    output = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, 128),)
    exp_overflow_opt_kernel[grid](input_tensor, output, n_elements, dim=-1, blockSize=128)
    # 归一化步骤
    sum_exp = output.sum(dim=-1, keepdim=True)
    return output / sum_exp
```

**坏实践（反例）**：适用于所有包含指数运算的场景（如Softmax、注意力分数计算），fp16/bf16下未做该处理易出现exp(>65504)返回Inf，910b2/910b3平台需强制启用该优化

**已知实例**：general_stability §raw (`experiences/general_stability/exp_stability_001.yaml`)
- 声称性能收益：数值稳定性提升100%

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../../ops/flash-attention-score-grad.md) — 同属优化知识，主题相关
- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 同属优化知识，主题相关
- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmaxV2 (MoE gating topk + softmax) on Ascend NPU](../../ops/moe-gating-top-k-softmax-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeGatingTopKSoftmax (MoE gating softmax + topK) on Ascend NPU](../../ops/moe-gating-top-k-softmax.md) — 同属优化知识，主题相关
- 实践案例: [PromptFlashAttention inference/prefill on Ascend NPU](../../ops/prompt-flash-attention.md) — 同属优化知识，主题相关
- 实践案例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 实践案例: [RingAttentionUpdate (Ring Attention output rescale merge) on Ascend NPU](../../ops/ring-attention-update.md) — 同属优化知识，主题相关
- 实践案例: [SparseFlashAttention inference (sparse KV) on Ascend NPU](../../ops/sparse-flash-attention.md) — 同属优化知识，主题相关
- 开发指南: [Fused Attention / Flash Attention 高性能算子](../../reference/op-class-attention.md) — 同属优化知识，主题相关
- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 同属优化知识，主题相关
- 开发指南: [归一化算子优化](../../reference/op-class-normalization.md) — 同属优化知识，主题相关
- 开发指南: [Softmax / LogSoftmax 算子优化](../../reference/op-class-softmax.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_attn_reduce_stage2_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_crossentropy_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: two pass algorithm](../compute-pipeline/exp_log_softmax_backward_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax](../compute-pipeline/exp_sdpa_infer_ascend_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: vectorized accumulation strategy](../compute-pipeline/exp_softmax_inner_002.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_non_inner_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: online softmax algorithm](../compute-pipeline/exp_softmax_online_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax](../memory-sram/exp_attn_sinks_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: online softmax algorithm](../memory-sram/exp_flashattn_fwd_001.md) — 同属优化知识，主题相关
- 相关主题: [长序列注意力分块计算 + 在线 Softmax（内存 O(S²)→O(S)）](../memory-sram/exp_fused_attention_002.md) — 同属优化知识，主题相关
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 aiv
- 相关主题: [UB 打满策略：调大分块吃满 192KB，微超则把 mask/exp 计算外提到调用侧](../memory-sram/ub-fill-and-compute-hoist.md) — 同属优化知识，主题相关
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 相关主题: [sqrt 前用 max(x, eps) 钳位防负数开方 NaN（eps 按精度选型）](exp_stability_002.md) — 主题相关：共享 aiv, general_stability, 数值稳定
- 相关主题: [关键计算步骤转 fp32 执行再转回原精度（MAE 降 80%，算力代价约 5%）](exp_stability_003.md) — 主题相关：共享 aiv, general_stability
- 相关主题: [累加器必须 fp32 初始化：fp16 累加 ≥1000 次漂移明显（MSE 降 90%）](exp_stability_004.md) — 主题相关：共享 aiv, general_stability
- 相关主题: [Online Softmax / Flash Attention 单遍稳定计算](online-softmax-flash-attention.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
