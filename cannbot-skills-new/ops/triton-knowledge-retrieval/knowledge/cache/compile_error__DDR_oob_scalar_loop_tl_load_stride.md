---
query_type: compile_error
query_keywords: [DDR_oob, scalar_loop, tl_load, stride, offset]
created: 2026-06-01
last_updated: 2026-06-01T09:40:00
hit_count: 2
context_summary: "scalar for循环中tl.load的指针偏移计算越界，需确保偏移在有效范围内"
top_units: [P0_scalar_offset_001, P1_mask_check_001]
stale: false
---

## 知识单元

**P0_scalar_offset_001** `[编译修复]` scalar循环中指针偏移越界
- **内容**: `tl.load(ptr + d_idx * stride)` 在 for 循环中，d_idx 从 0 到 BLOCK_D//2，但 stride 可能很大导致越界。
  - 正确做法: 确保 `d_idx * stride` 不超过张量边界，或用 mask 保护
  - 错误: `tl.load(gkc_base + d_idx * stride_gkc_d)` — 无 mask 保护
  - 正确: `tl.load(gkc_base + d_idx * stride_gkc_d, mask=(d_idx < half_d), other=0.0)`
- **来源**: compile_error 经验

**P1_mask_check_001** `[调试知识]` 所有 scalar load 必须有 mask
- **内容**: Ascend 硬件对越界访问敏感，即使条件为 False 的分支也可能触发 MTE 错误。
  - 推荐: 所有 tl.load/tl.store 都必须有 mask 保护
  - 推荐: 避免在 kernel 中使用 Python for 循环做 scalar 访问
- **来源**: triton-ascend-fundamentals.md
