---
query_type: compile_error
query_keywords: [runtime_error, acl_stream_sync, moe_gating, topk_softmax, int64_offset]
created: 2026-06-10
last_updated: 2026-06-10T11:47:00
hit_count: 1
context_summary: "MoE Gating TopK Softmax kernel 在 num_experts > BLOCK_SIZE 时触发 ACL stream synchronize failed 507035"
top_units: [P0_int64_offset_001, P1_mask_boundary_001, P1_ptr_dtype_001]
stale: false
---

## 检索问题（Problem Statement）

**错误症状**：Ascend Triton kernel 运行时崩溃，报错 `RuntimeError: ACL stream synchronize failed, error code:507035`

**触发场景**：
1. num_experts > BLOCK_SIZE 时，kernel 执行后 NPU 进入错误状态
2. 前6个 case（num_experts <= BLOCK_SIZE）通过，从第7个 case（num_experts=256 > BLOCK_SIZE=128）开始批量失败
3. 错误在后续 case 的 `x.to(device)` 时暴露（前 case 的异步设备错误残留）

**根本原因分析**：
- `offsets`（`tl.arange` 产生 `tl.int32`）与 `global_max_idx`（kernel 参数可能推断为 `tl.int64`）类型不匹配
- `offsets == global_max_idx` 比较和 `ptr + global_max_idx` 指针偏移在 Ascend 上可能因 int64/int32 混用触发未定义行为
- `topk_values_ptr.dtype.element_ty` 在 Ascend Triton 上访问指针 dtype 属性可能不支持
- BLOCK_SIZE 固定为128，当 num_experts 很小时，mask 外的偏移计算可能被编译器预计算导致越界

## 知识单元列表

### 核心修复知识单元

**P0_int64_offset_001** `[阻塞修复]` kernel 参数 int64 推断导致指针偏移类型不匹配
- **内容**: Triton kernel 标量参数默认可能推断为 `tl.int64`，而 `tl.arange` 产生 `tl.int32`。`offsets == global_max_idx` 比较和 `ptr + global_max_idx` 偏移可能因类型不匹配触发 Ascend 后端错误。
  - 错误: `is_target = (offsets == global_max_idx)` 其中 offsets=int32, global_max_idx=int64
  - 修复: 统一使用 `tl.int32` 类型，避免混合 int32/int64
  - 修复: `global_max_idx` 初始化为 `0`（Python int，Triton 推断为 int32）而非 `num_experts`
  - 修复: 用布尔标志 `has_found_max` 替代哨兵值 `num_experts`
- **来源**: ascend_patterns.md + runtime 分析

**P1_mask_boundary_001** `[重要修复]` BLOCK_SIZE > num_experts 时 mask 外的偏移预计算
- **内容**: Ascend 编译器可能在 mask 求值前预计算完整偏移量。当 BLOCK_SIZE=128 > num_experts=3 时，`ptr + offsets` 中 offsets=127 的地址被计算，即使 mask=False。
  - 修复: 确保分配的 tensor 宽度 >= BLOCK_SIZE，或在 host 侧根据 num_experts 选择更小的 BLOCK_SIZE
  - 修复: 使用 `triton.next_power_of_2(num_experts)` 作为 BLOCK_SIZE（需在 host 侧编译多个 kernel 变体，或改用固定小 BLOCK_SIZE）
- **来源**: compile_error__DDR_out_of_range_tl_load_mask_boundary.md

**P1_ptr_dtype_001** `[重要修复]` `ptr.dtype.element_ty` 在 Ascend 上访问可能不支持
- **内容**: `global_max.to(topk_values_ptr.dtype.element_ty)` 中访问指针的 `dtype.element_ty` 属性在 Ascend Triton 上可能返回未定义类型或触发编译错误。
  - 修复: 在 host 侧传入输出 dtype 信息，或根据输入 dtype 显式指定 `.to(tl.float16)` / `.to(tl.bfloat16)` / `.to(tl.float32)`
  - 修复: 使用 `tl.constexpr` 参数 `OUT_DTYPE` 控制输出类型
- **来源**: API 兼容性分析

**P1_scalar_cast_001** `[重要修复]` `(-inf).to(tl.int32)` 未定义行为
- **内容**: TopK 索引查找中 `block_max_idx = (-max_neg_score).to(tl.int32)`，当空 block 时 `max_neg_score = -inf`，`-(-inf) = inf`，`inf.to(tl.int32)` 产生未定义行为。
  - 修复: 先检查 `has_max_in_block`，仅在有效时用 `tl.where` 赋值索引
  - 修复: 用 `tl.zeros([1], tl.int32)` 替代 `.to(tl.int32)` 的转换路径
- **来源**: numerical stability 分析
