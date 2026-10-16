---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: 4.2
title: 迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）
description: 910b2/910b3 + CANN 8.5.0 上从第三方仓库（fla/vllm 等）迁移 kernel 时须删除不兼容参数：limit_auto_multi_buffer_of_local_buffer='no-limit'（未实现）、sync_solver=True（不支持）、tl.range num_stages（依赖 TMA，Ascend USE_TMA=False），保留会编译报错/警告
tags: [编译器约束/hints, linear-algebra-operations, solve_tril]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 迁移 kernel 须按 CANN 版本审计 launch 参数（8.5.0 删 no-limit/sync_solver/num_stages）

【优化维度】：编译器约束/hints
【适用条件】：平台 910b2/910b3 ｜ CANN 8.5.0（Triton-Ascend 对应版本）｜ dtype 全部 ｜ shape 任意（nd）
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3 + CANN 8.5.0；失效边界：本清单绑定 CANN 8.5.0——更高版本若实现 multi-buffer no-limit / sync_solver / TMA，对应参数可恢复；USE_TMA=False 是 910b + 8.5.0 的实测状态，平台或版本变化后 num_stages 失效性须重新确认；删除这些参数无性能影响（本来就不生效），只消除编译警告/错误；其他 CANN 版本未验证（unverified)）

### 原则
- **cann incompatible parameter removal**：CANN 不同版本支持的编译参数不同。以下参数在 CANN 8.5.0 上不支持或不兼容，应从 kernel launch 或 wrapper 中删除：`limit_auto_multi_buffer_of_local_buffer='no-limit'`（CANN 8.5.0 未实现）、`sync_solver=True`（CANN 8.5.0 不支持同步 solver）、`num_stages=2`（TMA 异步拷贝依赖，USE_TMA=False 时无效）。保留这些参数可能导致编译错误或警告。
- **version aware parameter audit**：从第三方仓库（如 fla）迁移 kernel 时，必须审计所有编译参数和 launch option 与目标 CANN 版本的兼容性。特别是 `tl.range(..., num_stages=N)` 中的 num_stages 依赖 USE_TMA 能力，而在 Ascend 上 TMA 通常不可用。
- **tma unavailable**：Ascend 910b 上 USE_TMA=False（CANN 8.5.0），TMA（Tensor Memory Access）异步拷贝不可用。依赖 TMA 的参数（num_stages、sync_solver 等）均为无效参数。

### 代码/骨架出处
- `problematic_params`:
```python
# 应删除的 CANN 8.5.0 不兼容参数：
# 1. limit_auto_multi_buffer_of_local_buffer="no-limit"  — CANN 8.5.0 未实现
# 2. sync_solver=True                                 — CANN 8.5.0 不支持
# 3. num_stages=2（在 tl.range 中）                     — 依赖 USE_TMA，Ascend 不可用

# 正确做法：从 wrapper 中删除不兼容参数
# merge_fn[NT, B * H](A=A, Ai=Ai, ...)  # 不含不兼容参数
```

**坏实践（反例）**：从 fla/vllm 等仓库原样搬运 kernel launch 参数到 Ascend → `limit_auto_multi_buffer_of_local_buffer='no-limit'`、`sync_solver=True`、`tl.range(num_stages=2)` 在 CANN 8.5.0 上未实现/依赖不可用的 TMA，触发编译错误或警告 → ✅ 正解即本条：逐项审计并删除不兼容参数，CANN 版本升级后重新审计

**已知实例**：solve_tril §raw (`experiences/solve_tril/exp_solve_tril_005.yaml`)
- 声称性能收益：无性能影响（参数在 CANN 8.5.0 上本就不生效），消除编译警告/错误
- 备注：此经验适用于所有从外部仓库（fla、vllm、flash-attention 等）迁移到 Triton-Ascend 的 kernel。迁移时务必逐项审计 launch option 和编译参数。

<!-- okf:related:start -->

# 相关

- 开发指南: [solve_tril 算子优化（三角矩阵求逆）](../../reference/op-class-solve_tril.md) — 主题相关：共享 solve_tril
- 相关主题: [autotune 必须探索 num_warps 与 num_stages](../autotuning/autotune-num-warps-stages.md) — 同属优化知识，主题相关
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 同属优化知识，主题相关
- 相关主题: [µs 级 kernel 禁用 autotune；num_stages/num_warps 在 Triton-Ascend 不生效](../autotuning/triton-ascend-microkernel-autotune-noise.md) — 同属优化知识，主题相关
- 相关主题: [Compute Optimization: redundant mask elimination by matrix property](../compute-pipeline/exp_solve_tril_001.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Compute Optimization: constexpr dual path full vs boundary](../compute-pipeline/exp_solve_tril_003.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [Scheduling Optimization: three tier hierarchical dispatch](../compute-pipeline/exp_solve_tril_004.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](../compute-pipeline/for-loop-tiling-ilp.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](../compute-pipeline/grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: preload extract slice eliminate hbm read](../memory-sram/exp_solve_tril_002.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../meta-skills/exp_solve_tril_006.md) — 主题相关：共享 linear-algebra-operations, solve_tril
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](constexpr-launch-specialization.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](exp_api_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](exp_api_002.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](exp_api_003.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](exp_api_004.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [移除确定性访存上的冗余 compile_hint（mayDiscretememaccess）](exp_ascend_compile_hint_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [Grid 必须是 ≤3 维 tuple 且各维乘积 ≤65535](exp_ascend_grid_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](exp_ascend_where_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](exp_constexpr_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [标量类型转换仅用 .to(type)，禁用类型构造函数](exp_scalar_cast_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](exp_slice_001.md) — 主题相关：共享 编译器约束/hints
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](exp_slice_002.md) — 主题相关：共享 编译器约束/hints

<!-- okf:related:end -->
