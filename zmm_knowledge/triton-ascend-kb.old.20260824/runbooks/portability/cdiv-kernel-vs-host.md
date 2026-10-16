---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: "4.2"
title: kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）
description: "@triton.jit kernel 内部必须用 tl.cdiv 计算 tile 数量，host 侧用 triton.cdiv 或整数除法，混用触发编译期报错"
tags: [cdiv, kernel_host_boundary, compile_error, triton_cdiv]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## CT-1 kernel 内 tl.cdiv / host 侧 triton.cdiv

【优化维度】：移植｜编译器约束
【泛化层级】：Triton通用（迁移条件：所有需要在 kernel 内计算 tile 数量的场景；无失效边界）

<约束：tl.cdiv 是 @triton.jit 内部编译期 tile 数计算原语；triton.cdiv 是 host 侧 Python 工具函数。两者作用域不重叠，禁止混用。与 [[constexpr-launch-specialization]] 同属编译器约束类。>

<违反后果：在 kernel 内调用 triton.cdiv 触发 `ValueError: Did you forget to add @triton.jit ?`，编译期报错，kernel 无法启动。>

<满足方式：
# kernel 内
@triton.jit
def kernel(...):
    num_tiles = tl.cdiv(total_elems, BLOCK_SIZE)   # tl.cdiv

# host 侧
grid = (triton.cdiv(N, BLOCK_SIZE),)               # triton.cdiv 或 (N + BLOCK_SIZE - 1) // BLOCK_SIZE
>

**已知实例**：GroupNormSwish §1 N6、AdaptiveInstanceNorm2DBackward §3.1 L1.4（`triton-op-generator/template/normalization.md`）

<!-- okf:related:start -->

# 相关

- 实践案例: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 主题相关：共享 compile_error
- 开发指南: [Legacy 知识单元类型与格式规范（triton-knowledge-retrieval）](../../reference/SPEC-Knowledge-Unit.md) — 同属优化知识，主题相关
- 开发指南: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
- 相关主题: [Triton API Quick Reference for Ascend Development](../../reference/triton-api-cheatsheet.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](../compute-pipeline/traj-general_perf-block_size.md) — 同属优化知识，主题相关
- 相关主题: [通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化](../compute-pipeline/traj-general_perf-scalar_elimination.md) — 主题相关：共享 cdiv
- 相关主题: [多维输入必须在 host 侧扁平化为 2D](../memory-sram/host-flatten-to-2d.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — 主题相关：共享 compile_error
- 相关主题: [Triton Kernel 通用语法避坑清单](../meta-skills/traj-general_api-syntax_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend 切片语法避坑：tl.arange 不是张量](../meta-skills/traj-general_ascend-syntax_pitfall.md) — 同属优化知识，主题相关
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend 调试排查典型问题聚合](../meta-skills/traj-general_debug-debugging.md) — 主题相关：共享 compile_error

<!-- okf:related:end -->
