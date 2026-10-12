---
schema_version: okf.v1
kind: runbook
type: perf_patterns
source_family: curated
title: Triton-Ascend 性能模式速查（官方调优手段）
description: 从 zh 官方性能指南提炼的可直接写入卡片的性能模式：care_padding、multibuffer 使能条件、grid 合并、指令并行 tiling。配合 backend_contracts.md（陷阱侧）使用——本卡是"应该主动做什么"，契约卡是"不要做什么"。
tags: [performance, care_padding, multibuffer, grid, tiling, instruction_parallel]
confidence: inferred
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-11T00:00:00Z"
---

# Triton-Ascend 性能模式速查

> [DOC] 数据源：zh `migration_guide/performance_guidelines.md`（NPU 高性能编程指南）。
> 陷阱侧见 `backend_contracts.md`；dtype 侧见 `dtype_support.md`。

## P-001 `care_padding=False`：去掉 mask 未覆盖区的默认填充 [DOC]

- **问题**：`tl.load(ptr, mask=m)` 未指定 `other` 时，NPU 为与 GPU 行为一致，先用 Vector 核
  把整个目标空间置 0（或 other 值），再 MTE2 搬入有效部分——Vector 与 MTE2 产生依赖，
  破坏搬运/计算并行。
- **写法**：未填充部分不影响后续计算时，`tl.load(ptr, mask=m, care_padding=False)`。
- **判定**：仅当 mask 外区域的值**不会被后续计算读取**时使用（如随后被同一 mask 保护）。

## P-002 multibuffer 使能三条件（不满足则乒乓失效）[DOC]

1. 搬运与计算之间无阻塞性数据依赖（否则 MTE 必须等 Vector）；
2. 算子内有多次数据加载或 tiling 循环（单次"搬入→算→搬出"无并行空间）；
3. UB 有富余（multi-buffer 增加 UB 占用；校核见 hardware_platforms.md）。

→ 卡片写 `multibuffer=True` 前须逐条确认三条件，否则选项静默无效。

## P-003 循环内 tiling 增加指令并行 [DOC]

- 单次顺序执行的 kernel（搬入→计算→搬出→结束）指令无法并行；用 `for` 循环 + BLOCK
  分块让多次迭代的搬运/计算流水起来，同时降低单次 UB 占用。
- 官方范式：`num_loop = tl.cdiv(N, BLOCK_SIZE); for i in range(num_loop): offs = base + i*BLOCK_SIZE`。
- ⚠️ 前提：分块后数学等价（归约类需确认累加顺序影响）。
- ⚠️ 注意：官方示例 kernel 内含 `%`，与 backend_contracts C-001 冲突，实际生成时
  按 C-001 改写为 `a-(a//b)*b`。

## P-004 大 grid 合并：`TRITON_ALL_BLOCKS_PARALLEL` [DOC]

- GPU 迁移来的算子 grid 分核数常远超 NPU 物理核数，多轮下发引入调度开销；
  优化第一步永远是**先检查 grid 分核数**。
- 手段：grid 固定为物理核数（persistent-core，见 backend_contracts C-011），
  或 `TRITON_ALL_BLOCKS_PARALLEL=1` + `auto_blockify_size`（顺序敏感 kernel 禁用，C-004）。

## P-005 对齐规避：借轴转置 [DOC]

- 尾轴不满足 32B（VV）/512B（CV）对齐时自动补齐导致性能恶化（如 (2048,3)）；
  可把对齐轴转置到低维计算、store 前转回；`numel % 256B == 0` 场景有官方"借轴转置"技巧
  （zh programming_guide/index.md 对齐注意条）。

## 卡片引用格式

P3「流水编排与可达性」或「Agent actions」节按需引用，如：
「load 的 mask 外区域不被消费 → 应用 P-001 care_padding=False」；
「单 kernel 无循环 → 按 P-003 增加内层 tiling 循环以满足 P-002 条件 2」。

## 维护记录

- 2026-09-11 初版：performance_guidelines 全量提炼（5 条）；dtype 相关条目归 dtype_support.md。
- 待办：P-001/P-003 的量化收益未实测，P4 回放时顺带采集。
