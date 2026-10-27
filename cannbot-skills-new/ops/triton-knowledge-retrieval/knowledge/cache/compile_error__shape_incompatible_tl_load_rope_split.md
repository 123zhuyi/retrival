---
query_type: compile_error
query_keywords: [shape_incompatible, tl_load, rope, split, half_dim]
created: 2026-06-01
last_updated: 2026-06-01T09:20:00
hit_count: 1
context_summary: "tl.load 中对已加载张量做切片时维度不兼容，需用指针偏移重新加载而非张量索引"
top_units: [P0_shape_fix_001, P1_rope_load_001]
stale: false
---

## 知识单元

**P0_shape_fix_001** `[编译修复]` 已加载张量不能用 tl.arange 做切片
- **内容**: `tl.load(tensor + offs_half)` 中 tensor 是已加载的 BLOCK_D 维张量，offs_half 是 BLOCK_D//2 维，维度不兼容。
  - 正确做法: 用指针偏移重新加载，而非对已加载张量做索引
  - 错误: `tl.load(grad_cos_term + offs_half)` — grad_cos_term 是 (BLOCK_D,) 张量
  - 正确: `tl.load(gkc_base + offs_half * stride)` — 从指针直接加载
- **来源**: compile_error 经验

**P1_rope_load_001** `[调试知识]` RoPE 半维度分别加载
- **内容**: 对 RoPE 的 half_dim 分割，应分别用两个 tl.load 从指针加载前半和后半，而非对已加载完整张量切片
  ```python
  # 正确: 从指针分别加载
  first_half = tl.load(ptr + offs_half * stride, mask=mask_half)
  second_half = tl.load(ptr + (offs_half + half) * stride, mask=mask_half)
  ```
- **来源**: triton-ascend-fundamentals.md
