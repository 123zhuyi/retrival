---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: autotuning
dimension: "5.1"
title: 累加/原地修改算子 autotune 必须每次重置输出 buffer
description: autotune 时每次重新 clone/初始化输出 buffer，否则重复累加同一 buffer 导致结果错误且性能失真
tags: [autotune, accumulate, buffer_reset, clone, acl_507035]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-1 autotune 输出 buffer 重置

【优化维度】：调优｜搜索空间
【泛化层级】：条件性（迁移条件：累加/原地修改类算子开 autotune；失效边界：纯覆盖写算子无须重置）

<原则：原地修改算子在 autotune 时重复累加同一 buffer，会导致结果错误且性能失真。必须每次重新 clone/初始化输出 buffer，或关闭 autotune。以目标维度（K/E/D）作为 autotune key，避免无关维度导致 config 错配。出现 ACL 507035 时优先检查是否未使用 UB/gather 或运行时 dim 分支。>

<骨架（占位名伪码）：
# autotune 每次重置
for config in autotune_configs:
    out_buf = torch.zeros_like(out)   # 每次重新初始化，否则重复累加
    kernel[grid](x, out_buf, ..., **config)
    # 评估 out_buf
# autotune key 用目标维度
@triton.autotune(configs=[...], key=['K', 'E'])  # 不是无关维度
>

**坏实践（反例）**：autotune 复用输出 buffer
```python
out_buf = torch.zeros_like(out)
for config in autotune_configs:
    kernel[grid](x, out_buf, ..., **config)  # out_buf 重复累加，结果错
```
后果：结果错误且性能失真。✅ 正解即每次重置 buffer 或关闭 autotune。

**已知实例**：index-computation G10（累加算子 autotune buffer 重置）

<!-- okf:related:start -->

# 相关

- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 主题相关：共享 autotune
- 相关主题: [Scheduling Optimization: autotune block m](../compute-pipeline/exp_gelu_bwd_001.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 预算与布局驱动的块大小选择](../memory-sram/tiling-budget-and-layout-driven-selection.md) — 主题相关：共享 autotune
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 主题相关：共享 autotune
- 相关主题: [波前调度 MatMul：对角遍历 + dot_pad_only_k + autotune tile 长宽比](../portability/exp_matmul_ascend_wavefront_001.md) — 同属优化知识，主题相关
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](autotune-num-warps-stages.md) — 主题相关：共享 autotune
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [Tiling 候选须过编译成本闸：超大分块档位编译爆炸直接剔除，host 公式分档可替代 autotune](tiling-candidate-compile-cost-gate.md) — autotune 使用陷阱族：本卡补编译成本闸维度
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](triton-ascend-microkernel-autotune-noise.md) — 主题相关：共享 autotune

<!-- okf:related:end -->
