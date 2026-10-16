---
query_type: compile_error
query_keywords: [negative_offset, safe_indexing, rope_rotation, tl_where, mask]
created: 2026-05-29
last_updated: 2026-05-29T16:48:00
hit_count: 1
context_summary: "Ascend Triton 运行时DDR越界：RoPE旋转中(offs±half_dim)导致负偏移或超界"
top_units: [A1, A2, A3]
stale: false
---

## 检索问题（Problem Statement）

**错误症状**：RoPE backward kernel 运行时 MTE DDR 越界

**触发代码**：
```python
k_val_first = tl.load(ks_base + (offs_d + half_dim) * stride_ks_d, mask=mask_d, other=0.0)
k_val_second = tl.load(ks_base + (offs_d - half_dim) * stride_ks_d, mask=mask_d, other=0.0)
```

**根本原因**：即使加了 mask，Ascend 后端仍可能计算 `(offs_d - half_dim)` 的负偏移，导致 MTE 地址越界。

## 知识单元列表

### 核心修复知识单元

**A1** `[安全索引]` **避免负偏移，使用 tl.where 重组数据**
- **方案**：不通过偏移加载另一半，而是分别加载两个半区再用 tl.where 拼接
- ```python
  # 安全方案：分别加载两个半区
  k_first = tl.load(ks_base + offs_d * stride_ks_d, mask=mask_d & (offs_d < half_dim), other=0.0)
  k_second = tl.load(ks_base + offs_d * stride_ks_d, mask=mask_d & (offs_d >= half_dim), other=0.0)
  # 然后重组
  ```

**A2** `[安全索引]` **使用两个独立 load 避免越界**
- ```python
  # 加载前半部分 [0:half_dim]
  offs_first = tl.arange(0, half_dim)
  k1 = tl.load(ks_base + offs_first * stride_ks_d)
  # 加载后半部分 [half_dim:D]
  offs_second = tl.arange(half_dim, D)
  k2 = tl.load(ks_base + offs_second * stride_ks_d)
  ```

**A3** `[kernel拆分]` **将 copy 和 compute 拆分为两个 kernel**
- 避免 kernel 内嵌套循环导致复杂偏移计算
