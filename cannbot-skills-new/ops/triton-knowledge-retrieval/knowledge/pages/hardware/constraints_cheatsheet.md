---
title: 硬件约束速查（生成算子强制检查）
category: hardware
tags: [constraints, checklist, 910b2, 910b3, ub, alignment]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# 硬件约束速查表

> **LLM 使用说明**：生成任何 Triton-Ascend kernel 之前，用此表检查所有约束项。违反任一项将导致编译错误、运行时崩溃或严重性能退化。

---

## 强制约束（违反 → 错误/崩溃）

| # | 约束 | 规则 | 示例 |
|---|------|------|------|
| C1 | **UB 容量限制** | 每个 tile 的数据量 ≤ 192KB（Atlas 800T/I A2） | fp16: BLOCK ≤ 98304 elements；fp32: ≤ 49152 elements |
| C2 | **Grid 大小上限** | grid_size ≤ AI Core 总数（≤ 65535） | `num_programs = min(n_rows, num_aicores)` |
| C3 | **512B 行宽对齐** | matmul 的块行宽必须是 512B 整数倍 | fp16: BLOCK_N 须为 512/2=256 的倍数；bf16 同 |
| C4 | **禁用 tl.swizzle2d** | 在 Ascend 上行为未定义，替换为 `grouped_launch_diagonal` | 见 [techniques/core_mapping](techniques/core_mapping.md) |
| C5 | **2D grid 需匹配拓扑** | 2D grid 的维度乘积 ≤ AI Core 数，且须对齐物理核拓扑 | 优先使用 1D grid，必要时咨询硬件文档 |

---

## 性能约束（违反 → 严重性能退化）

| # | 约束 | 症状 | 解决方案 |
|---|------|------|---------|
| P1 | **int64 CMP/ADD → scalar fallback** | `aiv_scalar_ratio` 异常高 | `.to(tl.float32)` 再做比较/加法 |
| P2 | **int32 CMP → scalar fallback** | 同上 | `.to(tl.float32)` |
| P3 | **无 care_padding=False 时 MTE2→Vector 依赖** | load/compute 串行，ILP 损失 | `tl.load(..., care_padding=False)` |
| P4 | **Block 数远多于核数（multi-round 调度）** | 核利用率低，启动开销大 | 设置 `TRITON_ALL_BLOCKS_PARALLEL=1` 或使用 persistent kernel |
| P5 | **非连续内存未预处理** | 向量化失效，访存效率低 | kernel 外先 `.contiguous()` |
| P6 | **大矩阵(M,N≥512)行主序 tiling** | L2 bank conflict → 性能下降 25-35% | 启用对角 tiling / diagonal core mapping |
| P7 | **fp16/bf16 accumulation 未升精度** | 数值溢出/精度问题 | 用 fp32 accumulator，最终 cast 回 fp16/bf16 |

---

## Tile Size 速查

| 数据类型 | 最大 BLOCK（单 tile 不超 UB） | 推荐 matmul BLOCK_M/N | 推荐 BLOCK_K |
|---------|------------------------|---------------------|-------------|
| fp16 (2B) | 98304 elements ≈ 313×313 | 128 或 256 | 256 |
| bf16 (2B) | 98304 elements ≈ 313×313 | 128 或 256 | 256 |
| fp32 (4B) | 49152 elements ≈ 221×221 | 128 | 128 或 256 |
| fp8 (1B) | 196608 elements | 256 | 256 |

> **注意**：matmul 的 L0 缓冲区同时持有 A、B、C tile，实际可用 UB 约为 1/3。以上为参考上限，实际需留 buffer。

---

## Autotune 搜索空间规范

```python
# matmul 标准搜索空间（满足 512B 行宽对齐 + UB 限制）
configs = [
    triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': 256, 'GROUP_SIZE': gs})
    for bm in [128, 256]
    for bn in [128, 256]
    for gs in [8, 16, 32]     # 至少 3 个不同 GROUP_SIZE
]

# elementwise 标准搜索空间
configs = [
    triton.Config({'BLOCK_SIZE': bs})
    for bs in [256, 512, 1024, 2048, 4096]
]
```

---

## 生成 Kernel 前的检查清单

```
□ BLOCK_SIZE * dtype_bytes ≤ 192KB / 3（为 A/B/C 三块留余量）
□ matmul 的 BLOCK_N * dtype_bytes 为 512 的整数倍
□ grid_size ≤ num_ai_cores（一般 910b2/b3 = 20 或 24）
□ 未使用 tl.swizzle2d（改用 grouped_launch_diagonal）
□ 所有 tl.arange() 结果做比较/加法前 .to(tl.float32)
□ tl.load() 中 care_padding=False（当 padding 不影响正确性时）
□ fp16/bf16 输入的归约/累加在 fp32 中进行
□ 大矩阵 M,N≥512 时启用对角 tiling
□ n_rows >> num_cores 时考虑 persistent kernel
□ BLOCK_M, BLOCK_N, BLOCK_K 均为 2 的幂次
```

---

## 核数获取方式

```python
import triton.language as tl
from triton.runtime import driver

device = torch.cuda.current_device()
props = driver.active.utils.get_device_properties(device)
num_sms = props["multiprocessor_count"]  # AI Core 总数

# 典型值：910b2 = 20 核，910b3 = 24 核
```

---

## 相关页面

- [hardware/ascend_910b](ascend_910b.md) — 完整硬件规格与编译选项
- [techniques/tiling](../techniques/tiling.md) — Tiling 策略详解
- [techniques/core_mapping](../techniques/core_mapping.md) — Core Mapping 与 GROUP_SIZE
- [techniques/data_types](../techniques/data_types.md) — Scalar Fallback 诊断与修复
- [techniques/ilp](../techniques/ilp.md) — care_padding 与流水线优化
- [techniques/persistent_kernels](../techniques/persistent_kernels.md) — Persistent kernel 模式
