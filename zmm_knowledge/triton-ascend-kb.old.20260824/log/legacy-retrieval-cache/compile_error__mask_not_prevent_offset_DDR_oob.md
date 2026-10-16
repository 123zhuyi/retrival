---
query_type: compile_error
query_keywords: [mask_not_prevent, offset, DDR_oob, tl_where, safe_load]
created: 2026-05-29
last_updated: 2026-05-29T16:52:00
hit_count: 1
context_summary: "Ascend Triton mask无法阻止偏移计算导致DDR越界，需用tl.where替代条件load"
top_units: [A1, A2]
stale: false
---

## 检索问题（Problem Statement）

**错误症状**：即使 tl.load 加了 mask，Ascend 后端仍计算完整偏移量导致 DDR 越界

**触发代码**：
```python
t1 = tl.load(ptr + (offs + half_dim) * stride, mask=mask & (offs < half_dim), other=0.0)
```

**根本原因**：Ascend 编译器可能在 mask 求值之前就计算了偏移量，导致越界地址被送入 MTE。

## 知识单元列表

### 核心修复知识单元

**A1** `[安全加载]` **使用 tl.where 替代条件 load**
- **方案**：先分别加载两个完整半区（都在有效范围内），再用 tl.where 选择
- ```python
  # 先加载前半区 [0:half_dim]
  offs_half = tl.arange(0, half_dim)
  first_half = tl.load(base + offs_half * stride)
  # 加载后半区 [half_dim:D]
  offs_second = tl.arange(half_dim, D)
  second_half = tl.load(base + offs_second * stride)
  # 重组
  ```

**A2** `[安全加载]` **使用两个独立 tl.load，范围不重叠**
- ```python
  # 加载 k1 (前半)
  k1 = tl.load(ks_base + tl.arange(0, half_dim) * stride)
  # 加载 k2 (后半)
  k2 = tl.load(ks_base + tl.arange(half_dim, D) * stride)
  ```
