---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.3"
title: 按 (维度, 模式, 采样比例) 多策略分派，禁止单 kernel 统所有路径
description: host 侧按特征分派到特化 kernel，通用 kernel 仅作兜底；禁止单一通用 kernel 统一处理所有路径
tags: [multi_strategy, dispatch, specialization, no_universal_kernel]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-09-14T00:00:00Z"
---

## OPT-1 多策略分派原则

【优化维度】：元技能｜设计决策
【泛化层级】：条件性（迁移条件：算子计算特征高度异构--不同 mode/比例/维度的最优数据访问模式完全不同；失效边界：特征单一的同构算子无须分派）

<原则：异构算子（如重采样有 nearest/bilinear/bicubic × 上/下采样 × 整数/分数倍）各场景最优数据访问模式完全不同。单一通用 kernel 用 gather 统一处理会丢失所有特化机会，坐标解码 overhead 极大。必须 host 侧按特征分派到特化 kernel，通用 kernel 仅兜底。>

<骨架（占位名伪码）：

```python
# host 侧 _route 分派
if mode == 'nearest' and H_out == 2*H_in and W_out == 2*W_in:
    nearest_2x_upsample_kernel[grid](...)      # 整数倍上采样：reshape broadcast
elif mode == 'nearest' and H_in == 2*H_out and W_in == 2*W_out:
    nearest_half_downsample_kernel[grid](...)  # 0.5x 下采样：strided load
else:
    general_kernel[grid](...)                  # 兜底通用路径
```

>

**坏实践（反例）**：单 kernel 统所有路径
```python
# 一个 universal kernel 用 gather 处理所有 mode 和比例
universal_resample_kernel[grid](x, output, mode, scale_h, scale_w, ...)
```
后果：坐标解码 overhead 极大，丢失特化机会；历史上出现过迭代 58 轮仍无法性能/精度双达标。✅ 正解即按特征多策略分派。

**已知实例**：tensor-transform G3、Interpolate L1.1（`triton-op-generator/template/tensor-transform.md`）；Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1）——serial persistent / chunked WY 双骨架 + host 成本模型 dispatch（`use_chunked = (n_seg*H <= 15) and (T >= 500)`，无 D2H 依赖），50 case 几何平均 22.25 超纯 serial 单骨架 19.91（+11.7%），大 T case 2.8-4.1×；分派条件由成本模型推导（chunked 2.85us·C·H + 5.5us·T/32·waves + 150us host vs serial 1.7us·T），见 [Gdn2Fwd 双骨架 dispatch 实测案例](../../ops/gdn2-fwd-chunked-wy-dispatch.md)（2026-09-14）

**已知实例（二次验证）**：Gdn2Fwd（varlen Gated DeltaNet 2 前向，ascend910b1）——同 guard `(not varlen) and (N*H <= 15) and (T >= 500)` 双骨架 dispatch 独立复现：geomean 31.2788（vs PyTorch eager），dispatch 命中 5 个 dense 长 T case 组几何 ×1.789（×1.54-2.10），见 [Gdn2Fwd 双骨架 dispatch 实测案例](../../ops/gdn2-fwd-chunked-wy-dispatch.md)（2026-09-16；证据：op_94_Gdn2Fwd_20260916_1619_2077/output/opt_iter_1/log.md）。


<!-- okf:related:start -->

# 相关

- 实践案例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../../ops/fused-floyd-attention.md) — 同属优化知识，主题相关
- 实践案例: [Gdn2Fwd（varlen Gated DeltaNet 2 前向）双骨架 host dispatch 实测案例：serial persistent + chunked WY](../../ops/gdn2-fwd-chunked-wy-dispatch.md) — Gdn2Fwd 双骨架 host 成本模型 dispatch 是本卡原则的实测案例
- 相关主题: [Moe Dispatch Skeleton](../../reference/skeleton-moe_dispatch.md) — 同属优化知识，主题相关
- 相关主题: [chunk 递推算子的依赖二分法与 Pingpong 下发调度](../compute-pipeline/chunk-recurrence-pingpong-dispatch.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: compilation specialization control](../compute-pipeline/exp_expand_serial_001.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: compile time dispatch](../compute-pipeline/exp_radix_convert_001.md) — 同属优化知识，主题相关
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](../compute-pipeline/exp_solve_tril_004.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: fused multi-tensor split and reshape](../memory-sram/exp_fused_qkvzba_split_001.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: dynamic specialization with aligned vector load](../memory-sram/exp_rope_forward_001.md) — 同属优化知识，主题相关
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [precompute dispatch（预反量化分派，gs<=64 或大 M 中等 N 启用）](precompute-dispatch.md) — 主题相关：共享 dispatch

<!-- okf:related:end -->
