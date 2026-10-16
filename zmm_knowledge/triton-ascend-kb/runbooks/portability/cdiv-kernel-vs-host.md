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
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 同属优化知识，主题相关
- 相关主题: [多维输入必须在 host 侧扁平化为 2D](../memory-sram/host-flatten-to-2d.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend kernel 调试排查清单](../meta-skills/debugging-checklist.md) — compile_error 映射表条目的具体契约
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 同属优化知识，主题相关
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](exp_constexpr_001.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
