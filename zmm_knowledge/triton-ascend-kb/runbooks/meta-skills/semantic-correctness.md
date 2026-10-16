---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: meta-skills
dimension: "6.2"
title: scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel
description: accumulate=False 重复索引实现 last-write-wins，越界检查加 mask，padding_idx 独立后置 kernel 清零
tags: [scatter_semantics, last_write_wins, boundary_check, padding_idx, independent_kernel]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## DG-3 scatter 语义正确性

【优化维度】：元技能｜精度修复链
【泛化层级】：条件性（迁移条件：scatter/index_put 类算子；失效边界：纯 gather 无写冲突）

<判定依据：scatter 算子三类语义陷阱--(1) 重复索引写冲突结果不稳定；(2) 越界访问触发 ACL 错误；(3) padding_idx 在热循环内判断破坏性能。必须分别处理。>

<处理规则：
1. last-write-wins：accumulate=False 且重复索引时，单 block 串行或两阶段实现，保证最后一个写胜出
2. 边界检查：index<0 或 index>=dim_size 加 mask；越界读返回 0，越界写加 mask 跳过
3. 排除位置：padding_idx 等通过独立后置 kernel 清零，不在热 scatter 循环内判断
>

**坏实践（反例）**：热循环内判断 padding_idx 或无边界检查
```python
for i in range(BLOCK):
    if idx[i] == padding_idx:   # 热循环内分支，破坏向量化
        continue
    if idx[i] < 0 or idx[i] >= dim:  # 散落判断
        ...
```
后果：写冲突结果不稳定；越界 ACL 错误；热循环分支破坏性能。✅ 正解即 last-write-wins + 边界 mask + 独立后置 kernel。

**已知实例**：index-computation G9（scatter/index_put 语义处理）

<!-- okf:related:start -->

# 相关

- 实践案例: [MoeTokenUnpermuteGrad (MoE token unpermute backward) on Ascend NPU](../../ops/moe-token-unpermute-grad.md) — 同属优化知识，主题相关
- 开发指南: [Gather/Scatter 算子优化](../../reference/op-class-gather_scatter.md) — 同属优化知识，主题相关
- 相关主题: [标量消除: block_ptr boundary_check 与 i32 比较定向消除](../compute-pipeline/exp_kda_bwd_intra_scalar_opt_001.md) — 同属优化知识，主题相关
- 相关主题: [内存访问优化：手动指针+向量化 mask 替代 block_ptr boundary_check](../memory-sram/exp_perf_011.md) — 同属优化知识，主题相关
- 相关主题: [Memory Access Optimization: padded binning for grouped scatter reduction](../memory-sram/exp_scatter_wgrad_moe_001.md) — 同属优化知识，主题相关
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../memory-sram/normalize-dim-to-last.md) — 同属优化知识，主题相关
- 相关主题: [Triton-Ascend strided-2/interleave 访存契约（三个正确性缺陷 + 寄存器重排正解）](../memory-sram/triton-ascend-strided2-interleave-contract.md) — 同属优化知识，主题相关
- 相关主题: [block_ptr boundary_check 引发标量降级，小 tile 改手动指针+向量化 mask](../portability/exp_ascend_blockptr_boundary_scalar_001.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
