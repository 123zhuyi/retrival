---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.1"
title: 跨 run benchmark 噪声鉴别纪律（framework 侧 3-7x 波动，speedup 跨 run 对比无效）
description: framework(torch eager) 侧延迟跨 run 波动 3-7x，speedup=framework/impl 跨 round 对比含噪声可致伪收益/伪回退；跨 round 结论必须以同 run impl 侧时延复核。
tags: [benchmark_noise, cross_run_variance, measurement_discipline, framework_jitter, pseudo_gain]
confidence: draft
status: draft
created_at: "2026-08-26T00:00:00Z"
updated_at: "2026-08-26T00:00:00Z"
---

## 跨 run benchmark 噪声纪律（来源：53_MobileViTAttention R2-R5 多次实测）

【泛化层级】：通用（迁移条件：所有跨 round/跨 run 的 speedup 比较；失效边界：同 run 内比较）

**机制（约束）**：benchmark.py 分别测 framework（torch eager）与 implementation（triton）两侧延迟求比。framework 侧跨 run 波动可达 **3-7x**（msprof 时延提取失败、NPU 设备状态、缓存），impl 侧相对稳定。后果：
- 伪收益：R2 的 case49 "+76%" 实为基线 framework 异常慢（10.60ms vs 正常 3.8ms，3x 跨 run 方差）；
- 伪回退/假平台期：同代码跨 round 复测 speedup 可差 16%（R4 1.5979 vs R5 复测基线 1.3361）。

**判别法**：同一代码两次 benchmark 的 impl 侧时延偏差 <5% 而 speedup 偏差 >5% → 噪声来自 framework 侧，speedup 差值不可信。

**复刻规则（纪律）**：
1. **跨 round 结论必须以同 run impl 侧时延复核**（speedup 对比只在同 run 内有效）；
2. round_result 的 yield_reason 涉及跨 round 比较时，注明口径（如"46-case 口径 1.415 vs R1 1.4064"）；
3. framework 侧异常 case（msprof 提取失败）应在几何平均中剔除并显式记录，不得静默计入；
4. 终局选最优 round 时，若候选 speedup 差 <10%，以 impl 侧总时延为决胜依据。

**失效边界**：同 run 内的 baseline vs optimized 对比（framework 同一进程同批测量，噪声相互抵消）。

<!-- okf:related:start -->

# 相关

- 相关主题: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](../autotuning/triton-ascend-microkernel-autotune-noise.md) — 同属优化知识，主题相关
- 相关主题: [Cross-kernel Optimization Transfer: token loop vs w loop decision](../compute-pipeline/exp_causal_conv1d_update_006.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
