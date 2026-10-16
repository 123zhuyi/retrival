---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.3"
title: GPU-source 迁移到 Ascend NPU 的 memory/index 改造检查清单
description: GPU/CUDA Triton 实现迁移到 Ascend 前必须完成的强制前置检查（设备/索引/累加/transpose/atomic/UB）
tags: [gpu_to_ascend, migration_checklist, cuda_to_npu, porting]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## DG-2 GPU->Ascend 迁移检查清单

【优化维度】：元技能｜设计决策
【泛化层级】：条件性（迁移条件：参考代码来自 GPU/CUDA Triton 实现；失效边界：原生 Ascend 实现无须此检查）

<判定依据：GPU 代码迁移后若 kernel 内仍出现 num_warps/cache_modifier/int64 索引下标/stride 欺骗 transpose/标量 atomic 内层循环，则迁移未完成。这些是强制前置条件，不应在 Phase 3/4 才修复。>

<检查清单（逐项核对）：
| 检查项 | 迁移规则 |
|---|---|
| 设备与启动 | cuda->npu；移除 num_warps/num_stages/cache_modifier/evict_last 等 GPU-only 选项 |
| tl.load/store 参数 | 禁用 boundary_check/padding_option 等 CUDA 参数；不越界时 care_padding=False |
| 索引类型 | 索引转 tl.int32；扁平偏移按需 int64；禁 fp64/uint（见 [[int32-index-cast]]） |
| 累加类型 | fp16/bf16 scatter/accumulate 用 fp32 中间 buffer（见 [[fp32-reduction-accumulator]]） |
| Block Pointer/转置 | 禁止交换 stride 假装 transpose；用 order=(...) 或 host permute().contiguous() |
| Memory 布局 | 转置收益须 > 2× 拷贝成本；无条件 contiguous 会抵消 gather 收益 |
| Atomic | 连续段优先向量 atomic_add（见 [[atomic-granularity]]）；标量 atomic 仅 fallback |
| UB 容量 | UB preload < 容量上限（fp32 ~32768 元素）；过大分块 BLOCK + BLOCK_SUB |
>

**判定口诀**：kernel 内若仍出现 num_warps/cache_modifier/int64 下标/stride 欺骗 transpose/标量 atomic 内层循环，则迁移未完成。

**已知实例**：index-computation G8（索引算子 GPU 迁移检查清单）

<!-- okf:related:start -->

# 相关

- 相关主题: [负载均衡分核：grid 对齐物理核数，核内循环替代 GPU 多维 grid](../compute-pipeline/load-balance-grid-sizing.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](debugging-checklist.md) — 迁移前置检查与调试排查清单互补
- 相关主题: [Profiling 驱动的瓶颈诊断与优化路由](profiling-bottleneck-checklist.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
