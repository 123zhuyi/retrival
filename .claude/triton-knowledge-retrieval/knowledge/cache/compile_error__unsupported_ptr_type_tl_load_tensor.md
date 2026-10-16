---
query_type: compile_error
query_keywords: [unsupported_ptr_type, tl_load, tensor, scalar_extract]
created: 2026-06-01
last_updated: 2026-06-01T09:25:00
hit_count: 1
context_summary: "tl.load 不能直接读取已加载的张量元素，需用标量提取或重新设计计算流"
top_units: [P0_scalar_extract_001, P1_tensor_element_001]
stale: false
---

## 知识单元

**P0_scalar_extract_001** `[编译修复]` 从张量提取标量值
- **内容**: `tl.load(tensor + idx)` 中 tensor 是张量而非指针，不支持。
  - 正确做法: 使用 `tl.get_element(tensor, idx)` 或 `tensor[idx]` 提取标量
  - 错误: `tl.load(grad_cos_exp + d_idx)` — grad_cos_exp 是张量
  - 正确: `grad_cos_exp[d_idx]` 或重新设计避免逐元素循环
- **来源**: compile_error 经验

**P1_tensor_element_001** `[调试知识]` 避免对已加载张量做指针操作
- **内容**: Triton Ascend 中，tl.load/tl.store 只接受指针参数。对已加载到寄存器的张量，应直接用张量运算而非指针运算。
  - 推荐: 用 tl.sum 做归约，避免手动 for 循环
  - 推荐: 用 tl.where 做条件选择
- **来源**: triton-ascend-fundamentals.md
