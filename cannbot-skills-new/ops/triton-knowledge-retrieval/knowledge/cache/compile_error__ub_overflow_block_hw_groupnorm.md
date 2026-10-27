---
query_type: compile_error
query_keywords: [ub_overflow, block_hw, groupnorm, tiling, local_buffer]
created: 2026-06-05
last_updated: 2026-06-05T14:45:00
hit_count: 1
context_summary: "GroupNorm kernel 编译时 UB 溢出，BLOCK_HW 过大导致 tl.arange 向量超出 UB 容量"
top_units: [C1, T1, P1]
stale: false
---

## 检索问题（Problem Statement）

GroupNorm kernel 编译失败，MLIRCompilationError: ub overflow, requires X bits while 1572864 bits available。

**触发上下文**：GroupNorm 代码生成后 verify 阶段，大 HW shape case 编译失败
**检索范围**：UB overflow / tiling basic block too large / multibuffer

---

## 知识单元列表

### P1 `[修复建议]` UB Overflow — Reduce BLOCK_SIZE
- **内容**：UB 容量 192KB = 1572864 bits。当 kernel 内 `tl.arange(0, BLOCK_SIZE)` 创建的向量过大，或多个临时变量同时占用 UB 时，会导致溢出。解决方案：
  1. 减小 BLOCK_SIZE，推荐值 {128, 256, 512, 1024}
  2. 避免在 kernel 内同时创建多个大向量
  3. 使用 for-loop tiling，将大 load 拆分为多个小 tile 循环
- **来源**：`pages/techniques/tiling.md`

### C1 `[硬件约束]` UB 容量上限
- **内容**：UB 192KB，fp16 最大 ~98304 elements；fp32 最大 ~49152 elements。GroupNorm 中 BLOCK_HW 应 ≤ 1024（保守）或 ≤ 2048（极限），确保多个临时变量同时存在时不溢出。
- **来源**：`pages/hardware/constraints_cheatsheet.md C1`

### T1 `[调优策略]` For-Loop Tiling 启用 ILP
- **内容**：将单个大 tile 拆分为循环处理多个小 tile，每个 tile  fit UB，编译器可自动做 multi-buffer / double-buffer 流水线优化。
- **来源**：`pages/techniques/tiling.md`

---

## 复用提示
- 含 `ub_overflow` + `block_size` → 直接复用 P1
- 含 `groupnorm` + `normalization` → 复用 C1, T1
