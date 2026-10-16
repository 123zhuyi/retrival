---
schema_version: okf.v1
kind: constraint
type: constraint_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.4
title: 逐元素算子小 shape 分裂专用 kernel 时，禁用"固定 BLOCK=阈值上限"的单块写法
description: kernel 分裂（小 shape 专用 kernel）若把专用 kernel 的 BLOCK 固定为阈值上限（如 numel<=8192 就 BLOCK=8192 单 program 无循环），小 numel case 被迫算满宽 tile，实测 geomean 0.5929 vs 基线 0.7293 严重劣化；未被证伪的变体：按 numel 细分多档小 BLOCK
tags: [kernel分裂, 小shape, 逐元素, general_ascend]
confidence: verified
status: draft
created_at: 2026-09-18T00:00:00Z
updated_at: 2026-09-18T00:00:00Z
---

## 逐元素算子小 shape 分裂专用 kernel 时，禁用"固定 BLOCK=阈值上限"的单块写法

【优化维度】：多核并行与负载均衡
【适用条件】：平台 910 系列（实测 Ascend910_9372）｜ 逐元素（elementwise）算子 ｜ dtype 全部 ｜ 混合 shape 评测集（含 numel≪阈值 的小 case）
【泛化层级】：条件性（失效边界：仅证伪"BLOCK 固定=阈值上限"这一种写法；按 numel 细分多档小 BLOCK（如 128/512/2048 分档）的专用 kernel **未被证伪**——参见 CLAUDE.md 负向经验粒度规则，不得以本记录跳过该变体）

### 原则
- **反模式**：host 侧 `if numel <= THRESH:` 路由到专用 kernel，专用 kernel 内 `BLOCK_SIZE: tl.constexpr = THRESH` 且 `grid=(1,)` 无循环——numel=100 的 case 也按 8192 宽 tile 计算+mask，有效算力利用率 1.2%
- **后果**：小 shape case 延迟不降反升，几何平均整体劣化 19%（0.5929 vs 0.7293）
- **满足方式**（未证伪变体，供重试）：专用 kernel 内仍保留 autotune 或按 numel 分档的小 BLOCK（128/512/2048），grid 仍按 `min(cdiv(numel, BLOCK), num_vectorcore)` 钳制

### 坏实践（反例）
```python
# ❌ 阈值上限直接当 BLOCK，单 program 无循环
if numel <= _SMALL_NUMEL:  # 8192
    _small_kernel[(1,)](x, y, numel, BLOCK_SIZE=_SMALL_NUMEL)  # numel=100 也算 8192 宽
```

### 已知实例
- GELU（activation，Ascend910_9372，50 case 混合 shape）案例：geomean 0.5929 vs 基线 0.7293 → 无提升回退（出处：`output/opt_iter_2/optimized_perf_result.json.speedup_vs_torch` 对比 `output/perf_result.json.speedup_vs_torch`）；verify 本身 50/50 通过（功能性无问题，纯性能反模式）；轨迹 `file:///home/w00836812/shixi_test02/cannbot-skills-new/plugins-official/triton-op-generator/triton_ascend_output/op_1_GELU_20260917_1827_8773/output/opt_iter_2/log.md`

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次起调](../autotuning/exp_perf_001.md) — 小 shape 分档 BLOCK 依据；含 65536 档证伪实例
- 相关主题: [逐元素算子 libdevice 超越函数 VECTOR bound 诊断链](exp_elementwise_erf_vector_bound_001.md) — 同一轨迹（GELU）

<!-- okf:related:end -->
