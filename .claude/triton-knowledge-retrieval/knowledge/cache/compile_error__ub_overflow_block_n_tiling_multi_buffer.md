---
query_type: compile_error
query_keywords: [ub_overflow, block_n, tiling, multi_buffer]
created: 2026-06-10
last_updated: 2026-06-10T14:35:00
hit_count: 1
context_summary: "Triton Ascend kernel UB 溢出，tiling block 过大导致多缓冲后超 192KB"
top_units: [C1_UB_192KB, P0_ub_overflow_001]
stale: false
---

## 检索问题

Triton Ascend kernel 编译时 UB 溢出：requires 3201024 bits while 1572864 bits available。

**触发上下文**: 编译阶段 BiShengHIR pipeline，BLOCK_N=64 导致 k_nope+k_rope+v+pv 总量超 UB
**检索范围**: UBOverflow / Tiling / MultiBuffer

---

## 知识单元列表

知识单元 C1 [硬件约束] UB 总用量 ≤ 192KB
  Ascend 910B UB = 192KB = 1572864 bits。
  来源: hardware/constraints_cheatsheet.md C1

知识单元 P0_ub_overflow_001 [调试建议] 减小 BLOCK_N 或分块 D 维度
  当 k_nope [BLOCK_N, D_NOPE] + v [BLOCK_N, D_V] + pv [BLOCK_N, D_V] fp32 同时存在时，
  多缓冲后极易超 UB。
  修复:
  1. 减小 BLOCK_N（如 64→16）
  2. 分块加载 V 的 D_V 维度，避免同时保留完整 [BLOCK_N, D_V]
  3. 避免显式存储大的中间变量，尽量在表达式中直接计算
  来源: compile_error/ub_overflow_common.md
