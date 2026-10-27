---
query_type: compile_error
query_keywords: [MTE_DDR_oob, int64, offset, rope_backward, tl_load, mask]
created: 2026-06-09
last_updated: 2026-06-09T09:35:00
hit_count: 1
context_summary: "KvCacheUpdateWithRopeBackward cos_sin kernel MTE越界：int64索引+条件load偏移计算导致DDR地址越界"
top_units: [P0_int64_offset_001, P1_safe_load_001, P2_block_split_001]
stale: false
---

## 知识单元

**P0_int64_offset_001** `[编译修复]` int64 索引导致 MTE 地址越界
- **内容**: kernel 参数中 stride/B/H/S/L/D 为 int64，与 int32 block 混合运算导致指针计算错误，Ascend MTE 报 DDR out of range。
  - 错误: `b * stride_b + h * stride_h + pos_l * stride_s + offs_d * stride_d` 中 stride 为 int64
  - 正确: host 侧将所有索引参数 `np.int32()` 化，kernel 内全 int32 运算
  - 错误: `cache_position` int64 tensor 直接传入 kernel
  - 正确: `cache_position.to(torch.int32)` 后再传入
- **来源**: hardware/constraints_cheatsheet.md C5 + compile_error 经验

**P1_safe_load_001** `[编译修复]` mask 不能阻止 Ascend 偏移越界计算
- **内容**: `tl.load(ptr + (offs + half_d) * stride, mask=(offs + half_d) < D)` 在 Ascend 上即使 mask=False 仍可能计算越界偏移，触发 MTE error。
  - 错误: 用 mask 保护条件 load，依赖 mask 阻止越界
  - 正确: 拆分为两个独立 load，范围不重叠且均在有效边界内
    ```python
    offs_1 = tl.arange(0, half_d)
    data_1 = tl.load(base + offs_1 * stride)
    offs_2 = tl.arange(0, half_d) + half_d
    data_2 = tl.load(base + offs_2 * stride, mask=offs_2 < D)
    ```
- **来源**: compile_error__mask_not_prevent_offset_DDR_oob.md

**P2_block_split_001** `[调试知识]` RoPE 旋转操作分块加载
- **内容**: RoPE 的 `k_rotated_half = [-k2, k1]` 需要前半和后半交叉访问， safest 方案是分别加载两个半区到独立 registers，再用 tl.where/拼接重组。
  - 前半区 load: `tl.arange(0, half_d)`
  - 后半区 load: `tl.arange(0, half_d) + half_d`
  - 避免任何 `offs + constant` 可能 >= D 的情况
- **来源**: triton-ascend-fundamentals.md
