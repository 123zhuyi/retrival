---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: memory-sram
dimension: "1.1"
title: preload + extract_slice 消除循环内同一数据重复 HBM 读取
description: 当矩阵块已在循环外通过 block_ptr 加载到寄存器后，循环内用 tl.extract_slice 提取子块，避免逐行重新从 HBM load。
tags: [preload, extract_slice, register_reuse, block_ptr, hbm_reread]
confidence: verified
status: verified
created_at: "2026-08-10T00:00:00Z"
updated_at: "2026-08-10T00:00:00Z"
---

## OPT-23 preload + extract_slice 消除循环内同一数据重复 HBM 读取

【优化维度】：内存｜访存效率
【泛化层级】：条件性（迁移条件：循环外已用 block_ptr 加载整个块，循环内需要按行/列/子块访问同一数据；失效边界：数据未在循环外加载，或 extract_slice 偏移无法表达）

<原则：Triton block tensor 不支持动态索引（`b_A[i, :]` 报错）。当矩阵块已在循环外加载到寄存器，循环内需要按行/列/子块访问同一数据时，应使用 `tl.extract_slice` 从寄存器提取，而非退化到 `tl.load(A + row_offset)` 重新从 HBM 读取。>

**核心问题**：

```python
# 循环外一次性预加载整个块到寄存器
b_A = tl.load(make_block_ptr(A, ..., block_shape=(BLOCK, BLOCK)))

# 错误：循环内逐行重新从 HBM 读取同一数据
for i in range(2, BLOCK):
    b_a = tl.load(A + i * stride)  # 重复 HBM 读！
```

**正例**：

```python
# 循环外：一次性预加载整个块到寄存器
b_A = tl.load(make_block_ptr(A, ..., block_shape=(BLOCK, BLOCK)))
neg_A = -b_A  # 预取反后的块

# 循环内：从寄存器提取，零 HBM 访问
for i in range(2, BLOCK):
    b_a = tl.extract_slice(neg_A, (i, 0), (1, BLOCK), (1, 1))
    b_a = tl.reshape(b_a, (BLOCK,))
    # ... 使用 b_a 计算 ...
```

**为什么不用 `b_A[i, :]`**？

Triton 报错 `ValueError('unsupported tensor index: int32[]')`——`i` 是运行时 int32，block tensor 不支持动态索引。`tl.extract_slice` 是 Triton 支持的寄存器块子集提取原语，offset 为运行时变量时生成动态偏移指令，开销远低于 HBM load。

**适用条件**：

- 数据已在循环外的寄存器中（通过 block_ptr 或一次性 load）
- 循环内需要按行/列/子块访问同一数据
- 循环次数 ≥ 2，预加载的额外寄存器操作可被摊还

**收益与权衡**：

| 场景 | 收益 |
|---|---|
| 大 grid（B≥4, T≥1024） | 显著（1.27x–1.37x），HBM load 节省累积放大 |
| 小 grid（B=1, T=512） | 微小（~1.01x），额外寄存器操作与省下的 load 几乎抵消 |
| 多对角块 | 每块独立预加载，各省循环体内 HBM 读取 |

**坏实践（反例）**：
```python
b_A = tl.load(make_block_ptr(A, ...))
for i in range(BLOCK):
    row = tl.load(A + i * stride)  # 重复读取已在寄存器中的数据
```
后果：同一数据多次 HBM 读取，带宽浪费。✅ 正解即 `tl.extract_slice`。

## 已知实例（P3 挂载）

### Experience YAML 反链
- `exp_solve_tril_002` — Memory Access Optimization （收益 1.01x-1.37x（BT=16, V4 vs V1）；1.25x-1.27x（BT=32, V4 vs V1，首次优化）；1.11x-1.13x（BT=64, V4 vs V1））
  - 此为 V4 优化的核心。与 V1（exp_solve_tril_001）组合后，ORIGINAL vs V4 累计加速比：BT=16 达 1.42x-2.54x，BT=32 达 1.26x-1.29x，BT=64 达 1.55x-1.57x。
- `exp_solve_tril_004` — Scheduling Optimization （收益 通过分治将 O(BT³) 求逆降为 O(BT²) 的递推 + O(BT) 个 16×16 矩阵乘法）
  - 此架构是算子设计层面的决策，非后期优化。16×16 原子单元的选择由 NPU 矩阵乘法硬件决定。

<!-- okf:related:start -->

# 相关

- 开发指南: [Ascend 910B2/910B3 Hardware Reference](../../reference/ascend-910b.md) — 同属优化知识，主题相关
- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md) — 同属优化知识，主题相关
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](../compute-pipeline/traj-solve_tril-compute.md) — 主题相关：共享 extract_slice
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](../portability/traj-general_ascend-hardware_constraint.md) — 主题相关：共享 block_ptr
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 主题相关：共享 block_ptr
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 同属优化知识，主题相关
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 block_ptr, register_reuse
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 block_ptr, extract_slice, preload, register_reuse
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 register_reuse

<!-- okf:related:end -->
