---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_perf
dimension: "2.1"
title: 通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化
description: 聚合 3 条 general_perf 在 Ascend 910B 上的 scalar elimination 经验，覆盖 i32 LT/GT/LE/GE 比较转 fp32 向量比较、min() 改 tl.where 向量化选择、tl.cdiv 改为编译期常量或乘加条件，收益 10%–35%。
tags: [general_perf, scalar_elimination, i32_compare, fp32_compare, min_to_where, cdiv, constexpr, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_perf/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# 通用标量消除：i32 比较转 fp32、min 改 tl.where 与 tl.cdiv 编译期化

## 概述

Ascend 910B 的 Vector 单元对整数运算支持有限：
- `int32` 的 `LT/GT/LE/GE` 比较会**标量降级**；
- `tl.cdiv` 包含运行时取余，而取余在 Ascend 上也会标量降级；
- Python `min()` 在 kernel 内会引入运行时分支和潜在的 i64 转换。

本卡 3 条 experience 分别给出三类标量降级的规避方案：
1. **i32 比较 → fp32 比较**：把 mask 生成中的整数比较改成浮点比较，走 Vector CMP；
2. **`min()` → `tl.where`**：用向量化条件选择替代运行时分支；
3. **`tl.cdiv` → 编译期常量或 `// + where`**：消除运行时除法取余。

**核心 trade-off**：
- 转 fp32 比较需要确保索引在 fp32 精确表示范围内（通常 SSM/attention 索引远小于 2²⁴）；
- `tl.where` 会计算两个分支，但避免了标量分支预测失败；
- 编译期化要求相关维度是 `tl.constexpr`，运行时变长场景需用 `// + where` 折中。

## 共同模式

### 1. i32 有序比较 → fp32 比较

**问题根因**：Ascend 910B 上，i32 的 `LT/GT/LE/GE` 比较会标量降级，只有 i32 的 `EQ/NE` 和 f16/f32 比较能走向量单元。`tl.arange` 默认 int64，直接生成 mask 会先触发 i64 比较降级。

**收益明显条件**：
- 代码中存在大量 `o_k < K`、`o_i >= j`、`(i_ti + o_i) < T_local` 等有序比较；
- 这些比较在热循环内或向量化 mask 生成中；
- `aiv_scalar_ratio` 高，且 hotspot 集中在 mask 生成处。

**实现要点**：
```python
# 反模式：i32 比较标量降级
m_k = o_k < K
m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T_local)

# 正模式：转 fp32 后比较
o_i_f = o_i.to(tl.float32)
o_k_f = o_k.to(tl.float32)
T_local_f = T_local.to(tl.float32)
K_f = K.to(tl.float32)
m_k = o_k_f < K_f
m_i_diag_qk = (o_i_f[:, None] >= o_i_f[None, :]) & ((i_ti + o_i_f[:, None]) < T_local_f)
```
- 比较前把 `tl.arange` 结果 cast 到 fp32；
- 把常量/标量也 cast 到 fp32；
- 比较结果仍是 boolean mask，可用于 load/store。

**常见坑**：
- 程序级单次分支（如 `if i_ti < T_local`）只执行一次，影响极小，无需转换；
- fp32 比较对超过 2²⁴ 的整数可能丢失精度；
- 不要把 mask 后续与 i32 运算混用导致类型不一致。

### 2. `min()` → `tl.where`

**问题根因**：Python `min()` 在 Triton kernel 中会产生运行时条件分支，且常伴随 i64 转换，导致标量开销和流水线停顿。

**收益明显条件**：
- `min()` 至少一个操作数是运行时值；
- `min()` 出现在循环边界或频繁执行的代码路径；
- profiling 发现标量分支是瓶颈。

**实现要点**：
```python
# 反模式：运行时 min() 分支
gn_offset = min(BC // 2, T_local - i_ti - 1)
for j in range(0, min(BC, T_local - i_t * BT - i_i * BC)):
    ...

# 正模式：tl.where 向量化条件选择
half_bc = BC // 2
remaining = T_local - i_ti - 1
gn_offset = tl.where(half_bc < remaining, half_bc, remaining)

loop_len_raw = T_local - i_t * BT - i_i * BC
loop_len = tl.where(BC < loop_len_raw, BC, loop_len_raw)
for j in range(loop_len):  # loop_len 仍需是 i32
    ...
```
- 用 `tl.where(cond, a, b)` 替代 `min(a, b)`；
- 循环边界变量仍需为 i32，必要时从 `tl.where` 结果提取或改用 `tl.minimum`（若后端支持向量）。

**常见坑**：
- 若 `min()` 的两个操作数都是 `tl.constexpr`，编译器会自动求值，无需替换；
- `tl.where` 的两个分支都会计算，若分支内有副作用需谨慎；
- `loop_len` 作为 `range` 边界必须是标量，不能是张量。

### 3. `tl.cdiv` → 编译期常量或 `// + where`

**问题根因**：`tl.cdiv` 在运行时执行整数除法和取余，而取余操作在 Ascend 上始终标量降级。

**收益明显条件**：
- 循环边界、tile 数量等计算中使用 `tl.cdiv`；
- `K/BK`、`V/BV` 等维度在编译期已知或整除；
- 运行时变长场景下仍需避免取余。

**实现要点**：
```python
# 反模式：运行时 tl.cdiv
for i_k in range(tl.cdiv(K, BK)):
    for i_v in range(tl.cdiv(V, BV)):
        ...

# 正模式 1：编译期常量循环边界（K、BK 为 constexpr）
NK = K // BK + (1 if K % BK != 0 else 0)
NV = V // BV + (1 if V % BV != 0 else 0)
for i_k in range(NK):
    for i_v in range(NV):
        ...

# 正模式 2：运行时变长场景替代
rows_in_chunk = T_local - i_t * BT
NC_local_full = rows_in_chunk // BC + tl.where(rows_in_chunk % BC > 0, 1, 0)
NC_local = tl.where(NC < NC_local_full, NC, NC_local_full)
```
- 编译期场景直接用 `//` 和条件表达式；
- 运行时场景用 `// + tl.where(余数 > 0, 1, 0)` 替代 `tl.cdiv`。

**常见坑**：
- 若所有相关维度都是 `constexpr`，优化效果最大；
- 运行时替代方案引入了额外 `tl.where`，需评估开销是否值得；
- 不要对已经是 `constexpr` 的除法用 `tl.cdiv`，直接写 `K // BK`。

### 4. 组合优化顺序

**问题根因**：上述三类标量降级可能同时存在，单独优化一项无法完全消除 scalar ratio。

**收益明显条件**：
- kernel 中同时有 i32 比较、`min()`、`tl.cdiv`；
- 已确认 scalar ratio 是主要瓶颈。

**推荐优化顺序**：
1. 先把 `tl.cdiv` 改为编译期常量或 `// + where`；
2. 把 i32 LT/GT/LE/GE 比较改为 fp32 比较；
3. 把 `min()` 改为 `tl.where`；
4. 最后 autotune tile 大小。

**常见坑**：
- 不要为了降 scalar 把矩阵乘逐元素展开，会丧失 Cube 加速；
- 每改一步都要重新 profiling；
- 保留 `tl.dot` 架构是底线。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_perf_012` | 35% | i32 LT/GT/LE/GE 比较转 fp32 比较，让 mask 生成走 Vector CMP；对 `tl.arange` 生成的索引比较尤其有效。 |
| `exp_perf_013` | 15% | `min()` 改 `tl.where` 向量化条件选择，消除运行时分支和 i64 转换；针对至少一个操作数为运行时值的场景。 |
| `exp_perf_014` | 10% | `tl.cdiv` 改为编译期常量计算或 `// + where` 替代，消除运行时整数除法取余的标量降级。 |

## 代表 Kernel

选择 `exp_perf_012` 作为代表（收益 35%），因为它覆盖的 **i32 比较转 fp32 比较** 是 Ascend 上最常见的标量瓶颈之一，且与 mask 生成密切相关。以下为该 experience 的代码片段：

```python
@triton.jit
def mask_with_fp32_compare(
    o_i, o_k,
    K: tl.constexpr,
    T_local: tl.constexpr,
    i_ti: tl.constexpr,
):
    # 反模式：i32 LT/GT/LE/GE → 标量降级
    # m_k = o_k < K
    # m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) \
    #             & ((i_ti + o_i[:, None]) < T_local)

    # 正模式：转 fp32 后比较 → 向量单元
    o_i_f = o_i.to(tl.float32)
    o_k_f = o_k.to(tl.float32)
    T_local_f = T_local.to(tl.float32)
    K_f = K.to(tl.float32)

    m_k = o_k_f < K_f
    m_i_diag_qk = (o_i_f[:, None] >= o_i_f[None, :]) \
                & ((i_ti + o_i_f[:, None]) < T_local_f)

    return m_k, m_i_diag_qk
```

对应 `exp_perf_013` 的 `min()` → `tl.where` 片段：

```python
    half_bc = BC // 2
    remaining = T_local - i_ti - 1
    gn_offset = tl.where(half_bc < remaining, half_bc, remaining)

    loop_len_raw = T_local - i_t * BT - i_i * BC
    loop_len = tl.where(BC < loop_len_raw, BC, loop_len_raw)
    for j in range(loop_len):
        ...
```

对应 `exp_perf_014` 的 `tl.cdiv` 替代片段：

```python
    # 编译期场景
    NK = K // BK + (1 if K % BK != 0 else 0)
    for i_k in range(NK):
        ...

    # 运行时变长场景
    rows_in_chunk = T_local - i_t * BT
    NC_local_full = rows_in_chunk // BC + tl.where(rows_in_chunk % BC > 0, 1, 0)
    NC_local = tl.where(NC < NC_local_full, NC, NC_local_full)
```

**关键实现细节**：
- `tl.arange` 默认 int64，用于比较前先 cast 到 fp32；
- `tl.where` 替代 `min()` 时，循环边界仍需保证是 i32 标量；
- `tl.cdiv` 在编译期场景下改 `K // BK + (K % BK != 0)`，运行时场景用 `// + where`；
- 这些优化通常组合使用，单独一项难以完全消除 scalar ratio。

## 已知实例

- `exp_perf_012` — Scalar Elimination（收益 35%）
  - i32 LT/GT/LE/GE 比较转 fp32 比较，让 mask 生成走 Vector CMP；对 `tl.arange` 生成的索引比较尤其有效。
- `exp_perf_013` — Scalar Elimination（收益 15%）
  - `min()` 改 `tl.where` 向量化条件选择，消除运行时分支和 i64 转换；至少一个操作数为运行时值时才有收益。
- `exp_perf_014` — Scalar Elimination（收益 10%）
  - `tl.cdiv` 改为编译期常量或 `// + where` 替代，消除运行时整数除法取余的标量降级。

## 关联知识

- 开发指南: [General_Perf 算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 主题相关：共享 910b2, 910b3, constexpr
- 对应指南: [归约算子优化](../../reference/op-class-reduction.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [扩展 Autotune 配置空间以覆盖多 Shape 最优调度](../autotuning/traj-general_perf-tuning_config.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [pergroup BLOCK_K 必须整除 group_size，选 min(group_size, 128)](../memory-sram/pergroup-blockk-align.md) — 同属优化知识，主题相关
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](../memory-sram/pow2-block-constexpr.md) — 主题相关：共享 constexpr
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [输出张量使用 empty_like 替代 zeros_like](../memory-sram/traj-general_perf-memory_init.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Ascend 循环适配：while 转 for 的静态与动态上限处理](../meta-skills/traj-general_api-loop_adapt.md) — 主题相关：共享 910b2, 910b3, constexpr
- 相关主题: [Ascend 条件表达式适配：tl.where 与 if-else 的选择](../meta-skills/traj-general_api-syntax_adapt.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Ascend 后端避免用 tl.where 动态计算内存偏移](../meta-skills/traj-general_ascend-syntax_adapt.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 主题相关：共享 910b2, 910b3, constexpr
- 相关主题: [避免 Stride 访问：优先连续化后一维遍历](../meta-skills/traj-general_perf-perf_pitfall.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Solve_Tril 优化决策：被排除的方向与根因](../meta-skills/traj-solve_tril-opt_decision.md) — 主题相关：共享 910b2, 910b3, constexpr
- 相关主题: [kernel 内用 tl.cdiv，host 侧用 triton.cdiv（禁止混用）](../portability/cdiv-kernel-vs-host.md) — 主题相关：共享 cdiv
- 相关主题: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../portability/constexpr-launch-specialization.md) — 主题相关：共享 constexpr
- 相关主题: [用 tl.constexpr 编译期消除运行时分支](constexpr-branch-elimination.md) — 主题相关：共享 constexpr
- 相关主题: [禁止在 kernel 内使用 continue/break，用 tl.where 向量分段](no-continue-break.md) — 同属优化知识，主题相关
- 相关主题: [Triton Block Size 通用调优原则](traj-general_perf-block_size.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [复杂算子拆分：降低编译与调度开销](traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [Element-wise 算子避免冗余 Reshape](traj-general_perf-redundancy_elim.md) — 主题相关：共享 910b2, 910b3, general_perf
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化](traj-ssm-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, fp32_compare, scalar_elimination

<!-- okf:related:end -->
