---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_ascend
dimension: "1.2"
title: Ascend 输出张量创建优化：empty/empty_like 替代 zeros/ones
description: 针对 Ascend 910B 上输出张量初始化开销，说明为什么优先用 torch.empty/torch.empty_like 创建输出张量，避免 torch.zeros/ones 的额外初始化开销，典型收益 8%。
tags: [general_ascend, memory_init, torch_empty, output_tensor, ascend, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_ascend/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Ascend 输出张量创建优化：empty/empty_like 替代 zeros/ones

## 概述

在 Triton kernel 调用前，通常需要在 host 侧创建输出张量。很多开发者习惯性地使用 `torch.zeros_like` 或 `torch.ones_like`，但这会在 Ascend 910B 上引入**额外的内存初始化开销**（约 5%–10%）。如果 kernel 随后会完整覆盖输出张量，这些初始化完全是浪费的。

本卡唯一的 experience `exp_tensor_create_001` 给出的规范是：**优先使用 `torch.empty` / `torch.empty_like`**。它们只分配内存、不填充值，且默认返回连续布局的张量。对元素数 ≥ 1e6 的大张量，可节省约 8% 的初始化时间。

**核心 trade-off**：
- `empty/empty_like` 快，但内存未初始化；
- 必须确保 kernel 会**完整覆盖**输出张量，否则可能读到脏数据；
- 对需要部分初始化的场景（如 padding 区必须为零），应显式初始化对应区域，而非全量 zeros。

## 共同模式

### 1. 优选 API：`torch.empty` / `torch.empty_like`

**问题根因**：`torch.zeros`/`torch.ones` 在分配内存后会启动一次 memset/fill kernel，对输出张量全部元素写 0 或 1。如果紧接着的 Triton kernel 会重写这些元素，这次 fill 就是无效工作。

**收益明显条件**：
- 输出张量会被 kernel 完整覆盖；
- 张量元素数较大（≥ 1e6）；
- 创建输出张量的操作在热路径上频繁发生。

**实现要点**：
```python
# 正确：使用 torch.empty_like，无初始化开销，默认连续
output = torch.empty_like(input_tensor)

# 或自定义形状
output = torch.empty(shape, dtype=dtype, device='ascend')
```
- `torch.empty_like` 会继承输入的 dtype、device 和 shape，并默认连续；
- `torch.empty` 可灵活指定 shape、dtype、device。

**常见坑**：
- 不要对未初始化张量做读取后参与计算，除非确认 kernel 已写入；
- `empty_like` 默认连续，但如果输入是 view/permuted，仍需确认输出布局是否符合 kernel 预期。

### 2. 禁用 API：`torch.zeros` / `torch.ones` / `zeros_like` / `ones_like`

**问题根因**：这些 API 会先初始化内存，Ascend 上 fill 操作虽然是带宽密集型，但对大 tensor 仍是可观延迟。

**收益明显条件**：
- 任何会被完整覆盖的输出张量创建；
- 大张量场景下效果更明显。

**实现要点**：
```python
# 错误：额外初始化开销
# output = torch.zeros_like(input_tensor)
# output = torch.ones_like(input_tensor)

# 正确：无初始化开销
output = torch.empty_like(input_tensor)
```
- 在代码审查中把 `zeros_like`/`ones_like` 用于输出张量视为反模式；
- 若业务逻辑确实需要初始化为 0/1，应显式说明原因。

**常见坑**：
- 某些框架模板或旧代码习惯使用 `zeros_like`，迁移时需检查；
- 不要为了“安全”无脑用 zeros，应依赖 kernel 的 mask 和 store 保证正确性。

### 3. 逻辑适配：确保未初始化内存被覆盖

**问题根因**：使用 `empty` 后，输出张量的值是不确定的。kernel 必须对所有有效位置执行 store，对无效位置（如 padding）也要有明确的 mask 控制。

**收益明显条件**：
- 使用 `empty` 创建输出张量；
- kernel 内部有 mask 或边界处理。

**实现要点**：
```python
@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = triton.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = data * 2
    tl.store(output_ptr + offsets, result, mask=mask)  # 只写入有效位置
```
- 用 `mask` 保证所有有效位置被写入；
- 无效位置保留未初始化值，但不会被下游使用。

**常见坑**：
- kernel 内有分支导致某些位置未 store；
- 下游算子读取了未初始化的 padding 区域；
- 对需要清零的特定区域，应单独用 `tl.store(..., 0.0, mask=...)` 处理。

### 4. 部分初始化场景的处理

**问题根因**：并非所有输出张量都能被完整覆盖。例如某些 padding 位置需要保持为 0，或某些区域需要作为默认值。

**收益明显条件**：
- 输出张量只有部分位置需要非零初值；
- 全量 zeros 初始化浪费带宽。

**实现要点**：
- 用 `torch.empty_like` 分配；
- 在 kernel 内对需要零值的区域显式 store 0；
- 或对默认值区域用 `tl.where`/`tl.store` 处理。

**常见坑**：
- 不要为了少量零值区域而全量 zeros；
- 显式初始化代码必须覆盖所有需要默认值的区域；
- 注意 mask 与默认值的组合逻辑。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_tensor_create_001` | 8% | host 侧创建输出张量优先用 `torch.empty`/`torch.empty_like`，避免 `zeros/ones` 的额外初始化开销；大张量（元素数 ≥ 1e6）效果更显著。 |

## 代表 Kernel

选择唯一的 `exp_tensor_create_001` 作为代表，展示输出张量创建规范：

```python
# 正确：使用 torch.empty_like（无初始化开销，默认连续）
def tensor_create_correct(input_tensor):
    output = torch.empty_like(input_tensor)

    # 错误示例（已注释）：使用 zeros/ones 会产生额外初始化开销
    # output = torch.zeros_like(input_tensor)
    # output = torch.ones_like(input_tensor)

    n_elements = input_tensor.numel()
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    elementwise_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE)
    return output

# 自定义形状时用 torch.empty
def tensor_empty_custom(shape, dtype=torch.float16):
    output = torch.empty(shape, dtype=dtype, device='ascend')
    return output

@triton.jit
def elementwise_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = triton.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = data * 2
    # 确保所有有效位置被覆盖，未初始化区域不会泄漏到下游
    tl.store(output_ptr + offsets, result, mask=mask)
```

**关键实现细节**：
- `torch.empty_like` 无初始化开销，且默认输出连续；
- kernel 内必须用 mask 保证所有有效位置被 store；
- 无效位置保留未初始化值，但下游不应依赖这些值；
- 自定义 shape 时用 `torch.empty(..., device='ascend')`。

## 已知实例

- `exp_tensor_create_001` — Memory Init（收益 8%）
  - host 侧创建输出张量优先用 `torch.empty`/`torch.empty_like`，避免 `zeros/ones` 的额外初始化开销；大张量（元素数 ≥ 1e6）效果更显著。代表 kernel 见上。

## 关联知识

- 开发指南: [Ascend NPU 特有语法与模式](../../reference/ascend-patterns.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [移除冗余 compile_hint 恢复编译器优化](../autotuning/traj-general_ascend-tuning_config.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 大 Shape 算子的交错循环负载均衡](../compute-pipeline/traj-general_ascend-load_balance.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 后端避免用 tl.where 动态计算内存偏移](../meta-skills/traj-general_ascend-syntax_adapt.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Triton-Ascend 切片语法避坑：tl.arange 不是张量](../meta-skills/traj-general_ascend-syntax_pitfall.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 硬件资源适配规则聚合](../portability/traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](../portability/traj-general_ascend-hardware_constraint.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [输出张量使用 empty_like 替代 zeros_like](traj-general_perf-memory_init.md) — 主题相关：共享 910b2, 910b3, memory_init, output_tensor

<!-- okf:related:end -->
