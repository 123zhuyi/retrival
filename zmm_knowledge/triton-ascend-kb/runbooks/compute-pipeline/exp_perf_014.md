---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: 2.1
title: '标量消除: tl.cdiv 编译期循环边界'
description: 'general_perf 场景下，标量消除: tl.cdiv 编译期循环边界，收益 10%'
tags: [AICore 单元利用/Scalar 削减, general, general_perf]
confidence: verified
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 标量消除: tl.cdiv 编译期循环边界

【优化维度】：AICore 单元利用/Scalar 削减
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3；失效边界：相关维度都是 constexpr 时收益最大；被除数为运行时值时替代方案（除法+tl.where）需评估额外开销是否值得；K/V/BK/BV 等维度在编译期不可知的动态 shape 场景不适用编译期方案）

### 原则
- **问题根因**：tl.cdiv在运行时执行整数除法和取余，取余操作在Ascend上始终标量降级，增加scalar开销
- **核心方案**：当除数和被除数相关参数为tl.constexpr时，在编译期计算循环边界（如NK = inputB // BK + tl.where(inputB % BK > 0, 1, 0)）
- **替代方案**：若被除数为运行时值，可用乘法和条件判断替代除法（如rows_in_chunk // BC + tl.where(rows_in_chunk % BC > 0, 1, 0)），仍比tl.cdiv更可控
- **适用条件**：循环边界、tile数量等计算，其中K/V/BK/BV等维度通常为constexpr
- **典型收益**：消除运行时整数除法+取余，降低Scalar单元占用

### 代码/骨架出处
- `Triton`:
```python
# 反模式：运行时 tl.cdiv
for i_k in range(tl.cdiv(K, BK)):
for i_v in range(tl.cdiv(V, BV)):
...

# 正模式：编译期常量循环边界
NK = K // BK + (1 if K % BK != 0 else 0)
NV = V // BV + (1 if V % BV != 0 else 0)
for i_k in range(NK):
for i_v in range(NV):
...

# 运行时变长场景替代
rows_in_chunk = T_local - i_t * BT
NC_local_full = rows_in_chunk // BC + tl.where(rows_in_chunk % BC > 0, 1, 0)
NC_local = tl.where(NC < NC_local_full, NC, NC_local_full)
```

**坏实践（反例）**：循环边界用运行时 `tl.cdiv(K, BK)` → 整数除法+取余在 Ascend 上始终标量降级，每个循环边界都付 Scalar 开销 → ✅ 正解即本条：维度为 constexpr 时编译期算好 `NK = K // BK + (1 if K % BK else 0)`；运行时变长场景用 除法+`tl.where` 替代。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_014.yaml`)
- 声称性能收益：10%
- 备注：当所有相关维度都是constexpr时，优化效果最大；若涉及运行时值，需评估替代方案引入的额外tl.where开销是否值得。

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general, general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general, general_perf
- 相关主题: [Memory Access Optimization: constexpr strides](../memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general
- 相关主题: [Add算子块大小动态调整优化经验](../memory-sram/exp_elem_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](../memory-sram/exp_elemwise_add_001_20260526.md) — 主题相关：共享 general
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](../memory-sram/exp_perf_002.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](../memory-sram/exp_perf_003.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](../memory-sram/exp_perf_004.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](../memory-sram/exp_perf_006.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](../memory-sram/exp_perf_008.md) — 主题相关：共享 general, general_perf
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](../memory-sram/exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](../memory-sram/exp_perf_015.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存访问优化：跨阶段寄存器复用消除重复 load](../memory-sram/exp_perf_016.md) — 主题相关：共享 general, general_perf
- 相关主题: [内存初始化优化：输出张量创建 API 选择](../memory-sram/exp_tensor_create_001.md) — 主题相关：共享 general
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../portability/cdiv-kernel-vs-host.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [Compute Optimization: mixed precision compute](exp_attn_bwd_q_001.md) — 主题相关：共享 aicore 单元利用/scalar 削减
- 相关主题: [性能避坑: 避免多维 stride 访问](exp_perf_007.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](exp_perf_012.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](exp_perf_013.md) — 主题相关：共享 aicore 单元利用/scalar 削减, general, general_perf

<!-- okf:related:end -->
