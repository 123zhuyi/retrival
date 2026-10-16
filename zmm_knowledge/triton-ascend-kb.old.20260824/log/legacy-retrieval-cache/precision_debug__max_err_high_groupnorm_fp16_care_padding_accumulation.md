---
query_type: precision_debug
query_keywords: [max_err_high, groupnorm, fp16, care_padding, accumulation]
created: 2026-06-05
last_updated: 2026-06-05T11:35:00
hit_count: 2
context_summary: "GroupNorm 精度验证大规模失败，care_padding=False 导致 out-of-bound 值污染累加"
top_units: [P1, P2, P3]
stale: false
---

## 检索问题（Problem Statement）

GroupNorm 精度验证大规模失败，fp16 pass_rate 仅 18%，fp32 pass_rate 99.34%。

**触发上下文**：精度验证失败，max_err 超阈值
**检索范围**：care_padding 导致 out-of-bound 值污染累加

---

## 知识单元列表

**P1** `[调试建议]` care_padding=False 与 mask 累加的安全性问题
- **内容**：`tl.load(..., mask=mask, other=0.0, care_padding=False)` 中，`care_padding=False` 消除了 MTE2→Vector 同步屏障，但 out-of-bound 位置的值不会被 `other` 填充，可能读取到 garbage 或相邻数据。当这些值参与 `tl.sum` 等累加计算时，会导致统计量（mean/var）计算错误。
- **修复**：在需要累加计算的场景中，**移除 `care_padding=False`**，或确保 BLOCK_SIZE 整除数据维度（无尾块）。
- **来源**：`pages/api/triton_api_cheatsheet.md`

**P2** `[调试建议]` fp16 精度敏感性与 fp32 累加
- **内容**：fp16 对数值误差更敏感。mean/var 计算必须在 fp32 中进行，但最终 cast 回 fp16 时仍可能引入误差。确保所有中间计算（包括 mean、var、inv_std）均在 fp32 中完成。
- **来源**：`pages/ops/normalization.md`

**P3** `[调试建议]` BLOCK_SIZE 与尾块处理
- **内容**：当 BLOCK_SIZE > 实际数据维度时，会产生尾块。尾块中的无效元素若被错误加载，会污染计算结果。建议：
  1. 选择 BLOCK_SIZE <= 数据维度
  2. 或选择 BLOCK_SIZE 为数据维度的约数
  3. 或使用 `boundary_check` 替代 mask
- **来源**：`pages/techniques/tiling.md`

---

## 复用提示

**可部分复用本缓存的 query_keywords 变体**：
- 含 `care_padding` + `mask` + `accumulation` → 精确命中
- 含 `precision` + `fp16` + `normalization` → 相似命中，复用 P2/P3
