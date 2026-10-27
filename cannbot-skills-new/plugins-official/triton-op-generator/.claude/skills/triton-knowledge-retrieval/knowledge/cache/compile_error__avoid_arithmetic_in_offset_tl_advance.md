---
query_type: compile_error
query_keywords: [avoid_arithmetic, offset, tl_advance, tl_make_block_ptr, safe_access]
created: 2026-05-29
last_updated: 2026-05-29T16:55:00
hit_count: 1
context_summary: "Ascend Triton 完全避免在tl.load偏移中使用算术运算，改用tl.make_block_ptr"
top_units: [A1, A2]
stale: false
---

## 检索问题（Problem Statement）

**错误症状**：任何包含算术运算的 tl.load 偏移都导致 DDR 越界

**根本原因**：Ascend 后端对动态/算术偏移的边界检查不完善

## 知识单元列表

### 核心修复知识单元

**A1** `[安全访问]` **使用 tl.make_block_ptr 替代手动偏移**
- ```python
  block_ptr = tl.make_block_ptr(
      base=ptr, shape=(S, D), strides=(stride_s, stride_d),
      offsets=(s, 0), block_shape=(1, BLOCK_D), order=(1, 0)
  )
  val = tl.load(block_ptr, boundary_check=(0, 1))
  ```

**A2** `[安全访问]` **完全避免 (idx ± const) 形式的偏移**
- 所有 tl.load 使用简单形式：`base + idx * stride`，其中 idx = tl.arange(0, N) 且 N ≤ 实际维度
