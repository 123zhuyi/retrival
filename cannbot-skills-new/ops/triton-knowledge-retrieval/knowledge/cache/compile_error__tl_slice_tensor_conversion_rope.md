---
query_type: compile_error
query_keywords: [tl_slice, tensor_conversion, rope, rotation]
created: 2026-05-30
last_updated: 2026-05-30T09:21:00
hit_count: 1
context_summary: "tl.slice 不是 Triton 有效 API，RoPE 旋转需用 tl.where 或索引计算实现"
top_units: [P0_slice_fix_001, P1_rope_index_001]
stale: false
---

## 知识单元

**P0_slice_fix_001** `[编译修复]` RoPE 旋转不使用 tl.slice/tl.concat
- **内容**: Triton Ascend 不支持 `tl.slice` 和 `tl.concat` 做张量切片/拼接。RoPE 的 `[-x2, x1]` 拼接应通过索引计算实现：
  - 方法1: 用 `tl.where(offs < HALF_DIM, -q_norm[offs + HALF_DIM], q_norm[offs - HALF_DIM])` —— 但注意 `tl.where` 做内存偏移在 Ascend 上可能低效
  - 方法2: 分别加载前半段和后半段，然后手动拼接（推荐）
  - 方法3: 用两个独立的 `tl.load` + `tl.store` 操作
- **来源**: triton-ascend-fundamentals.md §2.7

**P1_rope_index_001** `[调试知识]` RoPE 负索引处理
- **内容**: RoPE 中 `-x2` 表示取反，应在 fp32 域完成。推荐代码模式：
  ```python
  offs = tl.arange(0, HEAD_DIM)
  half = HEAD_DIM // 2
  # 前半段取反后半段
  rot_vals = tl.where(offs < half, -q_norm[offs + half], q_norm[offs - half])
  ```
  但注意 `tl.where` 用于值选择而非偏移选择更安全。
- **来源**: compile_error 经验
