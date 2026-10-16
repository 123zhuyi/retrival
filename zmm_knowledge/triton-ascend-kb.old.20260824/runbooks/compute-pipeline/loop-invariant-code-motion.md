---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: compute-pipeline
dimension: "2.1"
title: 循环不变量外提（LICM）减少重复地址计算
description: 将循环内重复的地址分量、stride 乘积、mask 提到循环外预计算，减少每轮迭代的标量开销。
tags: [licm, loop_invariant, pointer_base, stride_precompute, scalar_reduction]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-17 循环不变量外提（LICM）减少重复地址计算

【优化维度】：计算｜Scalar 削减
【泛化层级】：Triton通用（迁移条件：循环体内存在不随迭代变化的地址计算或 mask；失效边界：编译器已自动完成 LICM 且手动外提无额外收益）

<原则：循环内重复计算 `base + start_offset + dim_offs * stride` 会引入大量标量乘加指令。显式把不变分量提到循环外，循环内只做 `+ i * stride` 的简单偏移，可显著降低标量开销。seq_len=2048 时约节省 4096 次乘加地址计算。>

**反模式：循环内重复计算不变地址分量**：

```python
for i in range(seq_len):
    ptr = x_ptr + start * stride_t + d_offs * stride_d + i * stride_t
    x = tl.load(ptr, mask=mask)
```

**正模式：循环不变量外提**：

```python
x_ptrs = x_ptr + start * stride_t + d_offs * stride_d  # 外提
y_ptrs = y_ptr + start * stride_t + d_offs * stride_d

for i in range(seq_len):
    x = tl.load(x_ptrs + i * stride_t, mask=mask)       # 仅简单偏移
    tl.store(y_ptrs + i * stride_y, acc, mask=mask)
```

**常见可外提项**：

- `BT_arange[:, None] * V` / `H*K` / `H*V` stride 预计算
- `BK_mask` / `BV_mask`（若 K/V 为 BK/BV 整数倍可完全消除）
- 基址指针 `base_ptr + batch_offset`
- 循环外已知的 `tl.arange` broadcast 结果

**与 block_ptr 的关系**：

`tl.make_block_ptr` 已经做了大量地址预计算，但当需要手动索引（如小 tile、频繁边界 tile）时，显式 LICM 仍然有效。两者不冲突：block_ptr 用于硬件 block transfer，LICM 用于减少循环内标量指令。

**坏实践（反例）**：
```python
for i_v in range(NV):
    p_v_new = v_new + BT_arange[:, None] * V + BV_arange  # BT_arange*V 每轮都重算
```
后果：标量指令数随循环次数线性增长。✅ 正解即 `base_v_new = v_new + BT_arange[:, None] * V` 提到循环外。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_perf_016` — 内存访问优化 （收益 15%）
  - 寄存器复用与寄存器压力之间存在权衡。若kernel寄存器压力已很高，barrier可能导致spill，此时复用无效甚至劣化。建议通过profiling验证MTE2变化。
- `exp_kda_bwd_wy_structure_opt_001` — 计算优化 （收益 30%）
  - 此bundle聚焦kernel结构层面的优化，与标量消除优化正交互。实施时建议先完成标量消除，再做结构优化，便于独立验证每步效果。

<!-- okf:related:start -->

# 相关

- 相关主题: [for-loop tiling 启用双缓冲/多缓冲流水线](for-loop-tiling-ilp.md) — 同属优化知识，主题相关
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
