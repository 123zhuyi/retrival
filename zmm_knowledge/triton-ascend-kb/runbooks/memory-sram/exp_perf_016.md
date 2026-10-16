---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: 1.1
title: 内存访问优化：跨阶段寄存器复用消除重复 load
description: general_perf 场景下，内存访问优化：跨阶段寄存器复用消除重复 load，收益 15%
tags: [访存效率, general, general_perf]
confidence: provisional
status: verified
created_at: 2026-08-21T00:00:00Z
updated_at: 2026-08-21T00:00:00Z
---

## 内存访问优化：跨阶段寄存器复用消除重复 load

【优化维度】：访存效率
【泛化层级】：条件性（迁移条件：硬件 910b2, 910b3，fp16/bf16/fp32，多阶段 kernel 中同一数据需跨 tl.debug_barrier() 复用的场景；失效边界：寄存器压力已高时 barrier 可能触发 spill，复用无效甚至劣化；barrier 后不再需要的数据无收益；是否 spill 取决于编译器实现，必须 profiling 验证 MTE2 变化）

### 原则
- **问题根因**：同一数据在不同计算阶段被重复加载，浪费HBM带宽和MTE2时间
- **核心方案**：通过寄存器复用跨阶段保持数据。在tl.debug_barrier()之前加载的数据若barrier后仍需使用，且编译器未强制spill，可直接复用
- **典型场景**：多阶段 chunked kernel 中，b_q 在 Phase 1.5 加载、Phase 2.5 再次需要同一块；b_b 在 Phase 1 加载、Phase 2.5 复用
- **注意事项**：tl.debug_barrier()是否导致寄存器spill取决于编译器实现，需实测验证复用是否有效
- **典型收益**：减少2-4次block_ptr load/chunk，降低MTE2开销

### 代码/骨架出处
- `Triton`:
```python
# Phase 1.5：加载一次 b_q
b_q = tl.load(p_q, mask=BC_BK_mask, other=0.0)
# ... 使用 b_q 计算 ...
tl.debug_barrier()

# Phase 2.5：直接使用已加载的 b_q，无需重新加载
# b_q 仍在寄存器中（若未 spill）
# ... 继续使用 b_q ...
```

**坏实践（反例）**：barrier 后重新加载同一块数据
```python
b_q = tl.load(p_q, mask=m, other=0.0)   # Phase A 加载
tl.debug_barrier()
b_q2 = tl.load(p_q, mask=m, other=0.0)  # Phase B 重复加载同一块
```
后果：同一数据重复进 GM，每 chunk 多 2-4 次 load，浪费 HBM 带宽与 MTE2 时间。✅ 正解：barrier 后直接复用寄存器中的 b_q，但需先用 profiling 确认未发生 spill——寄存器压力高的 kernel 复用可能无效甚至劣化。

**已知实例**：general_perf §raw (`experiences/general_perf/exp_perf_016.yaml`)
- 声称性能收益：15%
- 备注：寄存器复用与寄存器压力之间存在权衡。若kernel寄存器压力已很高，barrier可能导致spill，此时复用无效甚至劣化。建议通过profiling验证MTE2变化。

<!-- okf:related:start -->

# 相关

- 相关主题: [块大小选型：2 的幂次 128/256/512 起调，blockSize×dtypeBytes 对齐 32B 且不超 UB](../autotuning/exp_perf_001.md) — 主题相关：共享 general, general_perf
- 相关主题: [autotune 配置空间须覆盖 num_warps(2/4/8)×num_stages(2/3/4) 与 tile 组合](../autotuning/exp_perf_017.md) — 主题相关：共享 general, general_perf
- 相关主题: [性能避坑: 避免多维 stride 访问](../compute-pipeline/exp_perf_007.md) — 主题相关：共享 general, general_perf
- 相关主题: [算子拆分优化: 按计算阶段拆分为简单 kernel](../compute-pipeline/exp_perf_010.md) — 主题相关：共享 general_perf
- 相关主题: [标量消除: i32 比较转 fp32 走向量单元](../compute-pipeline/exp_perf_012.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: min() 运行时分支改 tl.where](../compute-pipeline/exp_perf_013.md) — 主题相关：共享 general, general_perf
- 相关主题: [标量消除: tl.cdiv 编译期循环边界](../compute-pipeline/exp_perf_014.md) — 主题相关：共享 general, general_perf
- 相关主题: [if/else 分支 dtype 不一致：改 masked load，bool 张量 host 侧转 int32](../meta-skills/exp_debug_triton_branch_type_mismatch_001.md) — 主题相关：共享 general
- 相关主题: [Ascend 后端禁用语法及替代写法（控制流/lambda/链式布尔/直接索引/负偏移）](../portability/exp_api_001.md) — 主题相关：共享 general
- 相关主题: [复杂嵌套条件下 tl.where 改 if-else 分支](../portability/exp_api_002.md) — 主题相关：共享 general
- 相关主题: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../portability/exp_api_003.md) — 主题相关：共享 general
- 相关主题: [动态上限循环：for + if + MAX_ITERS 编译期上界替代 while](../portability/exp_api_004.md) — 主题相关：共享 general
- 相关主题: [禁止用 tl.where 动态计算内存偏移，改 if-else 静态分支](../portability/exp_ascend_where_001.md) — 主题相关：共享 general
- 相关主题: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../portability/exp_constexpr_001.md) — 主题相关：共享 general
- 相关主题: [禁用 Python 切片语法，改用 get_element/extract_slice/insert_slice](../portability/exp_slice_001.md) — 主题相关：共享 general
- 相关主题: [禁止对 tl.arange 索引表达式调用 get_element](../portability/exp_slice_002.md) — 主题相关：共享 general
- 相关主题: [大 shape 张量连续分块访存（缩小 grid、单核处理连续段）](exp_ascend_large_shape_block_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: kv tile reuse](exp_attn_bwd_hstu_001.md) — 主题相关：共享 访存效率
- 相关主题: [Memory Access Optimization: constexpr strides](exp_causal_conv1d_varlen_opt_001.md) — 主题相关：共享 general, 访存效率
- 相关主题: [Add算子块大小动态调整优化经验](exp_elem_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [带 alpha 缩放的逐元素 add 访存与块大小优化](exp_elemwise_add_001_20260526.md) — 主题相关：共享 general, 访存效率
- 相关主题: [内存访问优化：2D 张量块指针访存（make_block_ptr）](exp_perf_002.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：非连续张量连续化 + 一维遍历](exp_perf_003.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：stride 参数一律传 tensor.stride() 原生值](exp_perf_004.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：contiguous() 一次性拷贝替代 stride 访问](exp_perf_006.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建优选 torch.empty_like](exp_perf_008.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [冗余操作优化：同形 element-wise 算子免 reshape](exp_perf_009.md) — 主题相关：共享 general_perf
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](exp_perf_011.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存访问优化：行优先连续加载 + tl.trans 寄存器转置](exp_perf_015.md) — 主题相关：共享 general, general_perf, 访存效率
- 相关主题: [内存初始化优化：输出张量创建 API 选择](exp_tensor_create_001.md) — 主题相关：共享 general, 访存效率

<!-- okf:related:end -->
