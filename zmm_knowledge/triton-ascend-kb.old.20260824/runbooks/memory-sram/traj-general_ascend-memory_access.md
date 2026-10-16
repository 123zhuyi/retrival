---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_ascend
dimension: "1.1"
title: Ascend 大 Shape 连续分块内存访问优化
description: 针对 Ascend 910B 大 shape 内存密集型算子，通过缩小 grid、让每个 grid 处理连续一段数据并向上对齐到 BLOCK_SIZE，提升缓存命中率，性能提升 12%。
tags: [general_ascend, memory_access, large_shape, contiguous_block, cache, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_ascend/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Ascend 大 Shape 连续分块内存访问优化

## 概述

对大 shape 张量（元素数 ≥ 1e6）做 element-wise 或连续分块 matmul 时，内存访问模式往往比计算本身更影响性能。该经验通过 **缩小 grid、让每个 grid 负责一段连续数据**，最大化缓存命中率：

- 每个 program 处理一段连续的 `[grid_start, grid_end)` 区间；
- `ELEMENTS_PER_GRID` 向上取整到 `BLOCK_SIZE` 的整数倍，保证循环边界整齐；
- grid 数量由总元素数和 `ELEMENTS_PER_GRID` 共同决定，通常远小于元素数 / BLOCK_SIZE。

与交错循环（`range(pid, M, CORE_NUM)`）不同，这种方式强调 **空间局部性**：单个核心在一段时间内只访问连续地址，L1/L2 cache line 复用率更高。适合对内存连续性敏感、但不需要跨核心共享数据的算子。

收益约 12%，优先用于大向量 element-wise、连续分块 matmul 等场景。

## 共同模式

### 1. 缩小 grid，扩大每个 program 的连续处理范围

传统做法把 grid 设为 `triton.cdiv(n_elements, BLOCK_SIZE)`，每个 program 只处理一个 block。当元素数很大时 grid 也变大，可能触发调度开销甚至 grid 上限。该方案把多个 block 合并成一段连续区间交给一个 program，减少 grid 数量。

### 2. ELEMENTS_PER_GRID 对齐 BLOCK_SIZE

```python
ELEMENTS_PER_GRID = triton.cdiv(n_elements, MAX_GRID_SIZE)
ELEMENTS_PER_GRID = triton.cdiv(ELEMENTS_PER_GRID, BLOCK_SIZE) * BLOCK_SIZE
```

向上取整保证每个 program 的循环次数为整数，且最后一次迭代仍能用完整向量宽度加载，避免尾部大量 mask 处理。

### 3. Kernel 内顺序遍历连续区间

```python
grid_start = pid * ELEMENTS_PER_GRID
grid_end = min(grid_start + ELEMENTS_PER_GRID, n_elements)
for block_start in range(grid_start, grid_end, BLOCK_SIZE):
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < grid_end
    data = tl.load(input_ptr + offsets, mask=mask)
```

由于 `offsets` 在循环内连续递增，编译器可以生成合并的 burst load，缓存预取也能更有效工作。

### 4. 与交错循环负载均衡方案互补

- **连续分块**：强调空间局部性，适合纯内存密集型、无跨核心依赖的算子；
- **交错循环**：强调负载均衡，适合行级独立处理、需要严格均摊到每个核心的场景。

实际选择应根据算子特征和瓶颈分析决定。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_ascend_large_shape_block_001` | 12% | 大 shape 算子通过缩小 grid、连续分块处理提升缓存命中率，相比交错循环提升 12%。 |

## 代表 Kernel

`exp_ascend_large_shape_block_001` 没有独立的 extracted kernel 文件，以下为 YAML 中的 inline 代码示例：

```python
@triton.jit
def large_vector_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    # 计算当前 grid 负责的连续数据范围
    grid_start = pid * ELEMENTS_PER_GRID
    grid_end = min(grid_start + ELEMENTS_PER_GRID, n_elements)

    # 分块处理连续数据（缓存友好）
    for block_start in range(grid_start, grid_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < grid_end

        data = tl.load(input_ptr + offsets, mask=mask)
        result = data * 2 + 1
        tl.store(output_ptr + offsets, result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get(
                "vector_core_num", 40
            )
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, input_tensor):
        n_elements = input_tensor.numel()
        BLOCK_SIZE = 256
        MAX_GRID_SIZE = self.VEC_CORE_NUM

        # 计算每个 grid 处理的元素数（向上取整到 BLOCK_SIZE 倍数）
        ELEMENTS_PER_GRID = triton.cdiv(n_elements, MAX_GRID_SIZE)
        ELEMENTS_PER_GRID = triton.cdiv(ELEMENTS_PER_GRID, BLOCK_SIZE) * BLOCK_SIZE

        # 计算实际 grid 数量
        grid_size = triton.cdiv(n_elements, ELEMENTS_PER_GRID)

        output_tensor = torch.empty_like(input_tensor)
        large_vector_kernel[grid_size,](
            input_tensor, output_tensor, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            ELEMENTS_PER_GRID=ELEMENTS_PER_GRID,
        )
        return output_tensor
```

## 已知实例

- `exp_ascend_large_shape_block_001` — 内存访问优化（收益 12%）
  - 适用于元素数 ≥ 1e6 的大 shape 场景，1D/2D/3D 张量；优先用于大向量 element-wise、连续分块 matmul 等内存密集型算子。

## 关联知识

<!-- okf:related:start -->

# 相关

- 对应指南: [Legacy 缓存读写协议（triton-knowledge-retrieval）](../../reference/SPEC-Retrieve-legacy.md) — 同属优化知识，主题相关
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 cache
- 相关主题: [NsaCompressWithCache (NSA KV compression) on Ascend NPU](../../ops/nsa-compress-with-cache.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [移除冗余 compile_hint 恢复编译器优化](../autotuning/traj-general_ascend-tuning_config.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [扩展 Autotune 配置空间以覆盖多 Shape 最优调度](../autotuning/traj-general_perf-tuning_config.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 算子的交错循环负载均衡](../compute-pipeline/traj-general_ascend-load_balance.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 相关主题: [形状驱动算子决策树](../meta-skills/shape-driven-operator-decision-tree.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 后端避免用 tl.where 动态计算内存偏移](../meta-skills/traj-general_ascend-syntax_adapt.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Triton-Ascend 切片语法避坑：tl.arange 不是张量](../meta-skills/traj-general_ascend-syntax_pitfall.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 硬件资源适配规则聚合](../portability/traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](../portability/traj-general_ascend-hardware_constraint.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 输出张量创建优化：empty/empty_like 替代 zeros/ones](traj-general_ascend-memory_init.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
