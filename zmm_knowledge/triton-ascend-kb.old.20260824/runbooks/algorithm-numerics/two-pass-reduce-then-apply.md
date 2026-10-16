---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: algorithm-numerics
dimension: "3.4"
title: 跨 tile 归约必须显式两阶段 reduce-then-apply
description: 归一化拆为 Pass1 reduce（统计量）+ Pass2 apply（用聚合统计量归一化），单 pass 无法获得全局统计量
tags: [two_pass_reduction, cross_tile, normalize, global_statistic]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-2 跨 tile 归约两阶段 reduce-then-apply

【优化维度】：算法｜跨 Block 归约
【泛化层级】：Triton通用（迁移条件：归一化/全局统计量需跨 tile 合并；失效边界：单 tile 内可完成全部归约的无须拆分）

<原则：归一化本质是"先归约再仿射"，apply 阶段必须拿到完整 mean/var 才能正确归一化；反向 grad_input 公式依赖全局 s1/s2/s3 之和，无法在 reduce 阶段一次性算出。允许两种实现：单 kernel fused（统计量存寄存器，无中间 HBM 流量）或双 kernel split（partial sums 写 HBM，仅跨 program 合并时用）。与 [[fp32-reduction-accumulator]] 配套使用。>

<骨架（占位名伪码）：
# Pass 1: reduce（统计量）
for tile_offset in range(0, total_elems, BLOCK_SIZE):
    x = tl.load(input_ptr + tile_offset, mask=mask).to(tl.float32)
    sum_acc += tl.sum(x, axis=0)
    sq_acc += tl.sum(x * x, axis=0)
mean = sum_acc / total_elems
var = (sq_acc - sum_acc * sum_acc / total_elems) / total_elems
rstd = 1.0 / tl.sqrt(var + eps)

# Pass 2: apply（用聚合统计量归一化）
for tile_offset in range(0, total_elems, BLOCK_SIZE):
    x = tl.load(input_ptr + tile_offset, mask=mask).to(tl.float32)
    x_norm = (x - mean) * rstd
    tl.store(output_ptr + tile_offset, x_norm, mask=mask)
>

**坏实践（反例）**：在单个 tile 内直接计算最终 grad_input（需所有 tile 的全局统计和，单 tile 信息不全）
```python
# 单 pass 试图直接出 grad_input，缺少全局 s1/s2/s3
grad_input = grad_output * weight / std   # 漏掉 grad_var、grad_mean 的全局归约项
```
后果：grad_input 公式缺项，verify 失败。✅ 正解即本条两阶段拆分。

**已知实例**：GroupNormSwish §1 N2（单 kernel fused）、AdaptiveInstanceNorm2DBackward §1 N2（双 kernel split）（`triton-op-generator/template/normalization.md`）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](../../ops/moe-finalize-routing-v2-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](../../ops/moe-finalize-routing-v2.md) — 同属优化知识，主题相关
- 实践案例: [MoeInitRoutingV2Grad (MoE init routing backward) on Ascend NPU](../../ops/moe-init-routing-v2-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteGrad (token permute backward) on Ascend NPU](../../ops/moe-token-permute-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenPermuteWithEpGrad (MoE token permute with EP backward) on Ascend NPU](../../ops/moe-token-permute-with-ep-grad.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermuteWithEp (MoE token unpermute with EP) on Ascend NPU](../../ops/moe-token-unpermute-with-ep.md) — 同属优化知识，主题相关
- 实践案例: [MoeTokenUnpermute (MoE token unpermute / gather+reduce) on Ascend NPU](../../ops/moe-token-unpermute.md) — 同属优化知识，主题相关
- 实践案例: [NsaCompressGrad (NSA compress backward) on Ascend NPU](../../ops/nsa-compress-grad.md) — 同属优化知识，主题相关
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 同属优化知识，主题相关
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 同属优化知识，主题相关
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](../memory-sram/ub-aware-block-adaptive.md) — 同属优化知识，主题相关
- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](atomic-add-cross-tile-reduction.md) — atomic_add 是 two-pass 双 kernel split 的跨 tile 归并原语
- 相关主题: [数学归一化与算法前置条件必须显式处理（FFT 归一化/方向交替/模归约）](math-normalize-precondition.md) — 同属优化知识，主题相关
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](traj-normalization-memory_access.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
