---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: ssm
dimension: "2.1"
title: SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化
description: 针对 Ascend 910B 系列上 SSM（状态空间模型）反向 kernel 中标量指令占比过高的问题，核心是通过消除 i64 地址计算、把 i32 比较改为 fp32 比较、用 tl.where 替代 min/max、以及减少 block_ptr boundary_check 来降低 scalar ratio，提升 Cube 利用率。
tags: [ssm, scalar_elimination, i64, i32, fp32_compare, boundary_check, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/ssm/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化

## 概述

SSM（State Space Model）类模型（如 Mamba、Gated Delta Rule、Flash Linear Attention 等）的反向传播 kernel 在 Ascend 910B 上经常面临 **标量指令占比过高** 的问题。高 scalar ratio 意味着大量地址计算、比较、分支在 Vector/标量单元上执行，而 Cube（矩阵乘）单元处于饥饿状态，导致 AIC MAC 利用率极低（可低至 2%）。

本组 2 条 experience 均来自 `chunk_kda_bwd` 系列反向 kernel 的实测优化：

- **`exp_kda_bwd_wy_scalar_opt_001`**：wy_dqkg_fused 反向 kernel，scalar ratio 优化的完整案例，收益 233%；
- **`exp_kda_bwd_intra_scalar_opt_001`**：intra 反向 kernel，进一步定位到 block_ptr boundary_check 和 i32 整数比较是标量瓶颈，收益 50%。

**核心 trade-off**：

| 优化手段 | 收益 | 代价/约束 |
|---|---|---|
| **i64 地址计算 → i32** | 最大，避免 int64 算术标量降级 | 需确保索引范围在 2^31 内；SSM 典型尺寸满足 |
| **i32 整数比较 → fp32 比较** | 显著降低 scalar ratio |  mask 逻辑需用 float 张量；注意 bool 与 float 转换 |
| **`min/max` → `tl.where`** | 消除运行时分支 | 代码略冗长；float 比较更友好 |
| **`tl.cdiv` → 编译期常量** | 避免运行时除法取余 | 要求 K/BK、V/BV 在编译期整除 |
| **减少 block_ptr boundary_check** | 降低标量开销 | 需手写 mask 并保证正确性；适合规则访问模式 |
| **保留 `tl.dot` 架构** | 维持 Cube 加速 | 不要为了降 scalar 而逐元素展开矩阵乘 |

**重要原则**：标量消除的底线是 **保留 `tl.dot` 架构**。把矩阵乘逐元素展开虽然能降低 scalar ratio，但会丧失 Cube 加速，最终性能严重回退。

## 共同模式

### 1. 消除 i64 地址计算

在 Triton/Ascend 后端中，涉及 `int64` 的地址偏移、索引计算会被标量化（scalar fallback），因为 Vector 单元对 64-bit 整数运算支持有限。

**根因**：
- `i_tg = i_b * NT + i_t`、`bos/eos` 等偏移若用 i64，会触发标量 ALU；
- 即使数值本身在 32-bit 范围内，只要表达式中出现 i64 类型，整个地址计算链就可能降级。

**优化方案**：
- 把所有索引和偏移显式保持为 `int32`；
- 避免从 Python host 端传入 i64 标量；
- 对 `tl.arange` 结果也 cast 到 i32 再做乘加。

**收益**：这是 `exp_kda_bwd_wy_scalar_opt_001` 中收益最大的单项优化，贡献了 233% 加速中的主要部分。

### 2. i32 整数比较 → fp32 比较

mask 生成中的 `i32 <`、`>`、`<=`、`>=` 比较在 Ascend 上可能走标量路径。把操作数 cast 到 fp32 再做比较，可以利用 Vector 单元的浮点比较指令。

**根因**：整数比较的标量降级在某些后端实现中比浮点比较更常见；浮点 SIMD 比较指令更成熟。

**优化方案**：
```python
# 优化前（i32 比较，可能标量化）
m_t = o_t < T

# 优化后（fp32 比较）
m_t = o_t.to(tl.float32) < T.to(tl.float32)
```

**注意事项**：
- 比较结果仍是 boolean mask，后续与 `&`、`|` 组合时类型需一致；
- 对负数的比较，fp32 和 i32 语义相同（因为 -1e9 仍小于 0），但要避免极大正整数超过 fp32 精度；
- SSM 中的索引通常不会达到 fp32 精度极限（2^24），安全。

### 3. `min/max` → `tl.where`

`tl.minimum`/`tl.maximum` 或 Python `min()` 在边界处理中可能引入运行时分支或标量路径。

**优化方案**：
```python
# 优化前
last = min(t_end, T) - 1

# 优化后
last = tl.where(t_end < T, t_end, T) - 1
```

**根因**：`tl.where` 是无分支的向量化选择，所有 lane 执行相同指令；`min`/`max` 在某些后端会被拆成比较 + 选择，甚至引入分支。

### 4. `tl.cdiv` → 编译期常量

当 `K % BK == 0`、`V % BV == 0` 时，`tl.cdiv(K, BK)` 可以改为编译期常量 `NK = K // BK`。

**根因**：`tl.cdiv` 包含运行时除法和取余，会被标量化；编译期常量可以直接展开为 `range(NK)` 循环。

**优化方案**：
```python
# 优化前
for i_k in range(tl.cdiv(K, BK)):
    ...

# 优化后（K、BK 为 constexpr 且整除）
NK = K // BK
for i_k in range(NK):
    ...
```

### 5. 减少 block_ptr boundary_check

`tl.make_block_ptr(..., boundary_check=(0, 1))` 由硬件处理边界检查，但会产生额外的标量指令用于设置 block transfer 描述符。在某些密集访问场景下，手动计算指针 + 向量化 mask 比 block_ptr 更省 scalar。

**根因**：
- block_ptr 适合规则、大步长的 2D 访问；
- 当访问模式复杂或需要频繁边界检查时，block_ptr 的元数据开销会超过收益。

**优化方案**（来自 `exp_kda_bwd_intra_scalar_opt_001`）：
```python
# 优化前：block_ptr store
p_dq2 = tl.make_block_ptr(dq2, ..., boundary_check=(0, 1))
tl.store(p_dq2, b_dq2)

# 优化后：手动指针 + mask
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)
m_ti = (i_ti + o_i).to(tl.float32) < T_local.to(tl.float32)
m_k  = o_k.to(tl.float32) < K.to(tl.float32)
BC_BK_mask = m_ti[:, None] & m_k[None, :]
p_dq2 = dq2_local + (i_ti + o_i[:, None]) * H * K + o_k[None, :]
tl.store(p_dq2, b_dq2.to(dq2.dtype.element_ty), mask=BC_BK_mask)
```

**注意事项**：
- 手写 mask 必须保证不越界；
- 仅在对性能敏感且访问模式不规则时采用；
- 规则的大块 2D 访问仍推荐 block_ptr。

### 6. 其他微优化

- **`other=0.0` 直接用于 `tl.load`**：避免 load 后再 `tl.where` 清零；
- **移除冗余 `tl.compile_hint`**：某些 hint 会限制编译器优化空间；
- **autotune 放在最后**：先消除标量瓶颈，再用 autotune 微调 tile 大小。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_kda_bwd_wy_scalar_opt_001` | 233% | wy_dqkg_fused 反向 kernel：消除 i64 地址计算、i32 比较转 fp32、`tl.cdiv` 改编译期常量、`min` 改 `tl.where`、冗余 hint 移除。Task Duration 从 ~169k μs 降至 ~50.7k μs。 |
| `exp_kda_bwd_intra_scalar_opt_001` | 50% | intra 反向 kernel：已正确用 i32 但 scalar 仍高，进一步把 block_ptr store/load 改为手动指针 + mask，所有 i32 比较改 fp32，`min` 改 `tl.where`。预期 AIV scalar 从 41.4% 降至 <20%，AIC scalar 从 31.4% 降至 <15%。 |

## 代表 Kernel

本组 experience 的原始 YAML 未附带独立 kernel 文件，以下用 `exp_kda_bwd_wy_scalar_opt_001` 的代码片段展示核心优化模式：

```python
# 1. i64 → i32：所有索引偏移保持 i32
i_tg = i_b * NT + i_t  # 确保 i_b, NT, i_t 都是 i32

# 2. i32 比较 → fp32 比较
m_t = o_t.to(tl.float32) < T.to(tl.float32)

# 3. tl.cdiv → 编译期常量（要求 K % BK == 0, V % BV == 0）
NK = K // BK
NV = V // BV
for i_k in range(NK):
    for i_v in range(NV):
        ...

# 4. min → tl.where
t_end = i_t * BT + BT
t_last = tl.where(t_end < T, t_end, T) - 1
```

对应 `exp_kda_bwd_intra_scalar_opt_001` 的手动指针 + mask 模式：

```python
# block_ptr store → 手动指针 + mask
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)
m_ti = (i_ti + o_i).to(tl.float32) < T_local.to(tl.float32)
m_k  = o_k.to(tl.float32) < K.to(tl.float32)
BC_BK_mask = m_ti[:, None] & m_k[None, :]
p_dq2 = dq2_local + (i_ti + o_i[:, None]) * H * K + o_k[None, :]
tl.store(p_dq2, b_dq2.to(dq2.dtype.element_ty), mask=BC_BK_mask)
```

## 已知实例

- `exp_kda_bwd_wy_scalar_opt_001` — Scalar Elimination （收益 233%）
  - wy_dqkg_fused 反向 kernel 标量消除完整案例。关键原则：保留 `tl.dot` 架构，定向消除导致标量降级的 i64/i32 操作；逐元素展开会降低 scalar ratio 但会丧失 Cube 加速，应避免。
- `exp_kda_bwd_intra_scalar_opt_001` — Scalar Elimination （收益 50%）
  - intra 反向 kernel 进一步分析：即使正确使用 i32，block_ptr boundary_check 和 i32 整数比较仍是主要标量根因。优化时需特别注意转置 dAqk/dAkk 加载的 mask 布局。

## 关联知识

- 开发指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 910b2, 910b3, ssm
- 对应指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 ssm
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 主题相关：共享 ssm
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 主题相关：共享 ssm
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, boundary_check
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [scatter 语义正确性--last-write-wins + 边界检查 + 排除位置独立 kernel](../meta-skills/semantic-correctness.md) — 主题相关：共享 boundary_check
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](../portability/traj-general_ascend-hardware_constraint.md) — 主题相关：共享 910b2, 910b3, boundary_check
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — 主题相关：共享 ssm
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化](traj-general_perf-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, fp32_compare, scalar_elimination
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, ssm

<!-- okf:related:end -->
