---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: general_ascend
dimension: "4.1"
title: Ascend 硬件资源适配规则聚合
description: 聚合 2 条 general_ascend 硬件适配经验：动态获取 AI Core 数量以避免设备同步损耗，以及大 block 内存超限时进行合法切分。
tags: [general_ascend, hardware_adaptation, core_num, block_split, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/general_ascend/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Ascend 硬件资源适配规则聚合

## 概述

Ascend 910B2/910B3 有明确的硬件资源限制：Vector Core 数、Cube Core 数、block 内存上限等。写 Triton kernel 时如果硬编码这些数值或用错获取时机，轻则性能下降（设备同步），重则编译失败或运行时异常。本卡聚合 2 条 general_ascend 的 hardware adaptation 经验，一条关于核心数选择，一条关于 block 内存超限切分。

## 共同模式

### 1. 动态获取 AI Core 数量

不同型号 Ascend 的 Vector/Cube Core 数不同：
- 910B2：VEC=48、CUBE=24（默认）
- 910B3：VEC=64、CUBE=32（默认）

如果 kernel 内部或 host 端硬编码 core 数，在另一型号上会导致：
- grid size 不匹配，产生隐式设备同步；
- 同步开销通常带来 10%–15% 性能损失。

正确做法：
- 通过 `driver.active.utils.get_device_properties` 等 API 在运行时获取实际 core 数；
- 用该值作为 grid size 或 loop stride；
- 设置合理的兜底值，防止 API 失败时 grid 为 0。

### 2. Block 内存超限切分

Ascend 对单个 block 的内存有硬性限制：
- `BLOCK_SIZE ≥ 65536` 会直接编译失败；
- block 内存超过缓存容量会导致访存效率下降 50% 以上。

优化做法：
- 在 host 端根据 shape 和硬件限制计算是否需要切分；
- 把大循环拆成多个小循环，每个 block 内存合法；
- 切分后尽量保持合并访问，避免引入随机寻址。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_ascend_core_num_001` | 规避 10-15% 同步损耗 | 动态获取 VEC/CUBE core 数，避免硬编码导致设备同步 |
| `exp_ascend_split_001` | 避免编译失败 + 恢复访存效率 | BLOCK_SIZE≥65536 时切分，防止内存超缓存 |

## 代表 Kernel

以下为体现核心数动态获取的示意代码：

```python
# host 端（伪代码）
import triton

def get_num_aicore():
    try:
        props = triton.driver.active.utils.get_device_properties()
        return props["num_cores"]  # 或类似字段
    except:
        return 24  # 910B2 兜底

def launch_kernel(x, y):
    num_cores = get_num_aicore()
    grid = (num_cores,)
    my_kernel[grid](x, y, num_cores=num_cores)

# kernel 端
@triton.jit
def my_kernel(x_ptr, y_ptr, num_cores: tl.constexpr):
    pid = tl.program_id(0)
    n = tl.num_programs(0)
    # grid-stride 循环，用运行时传入的 num_cores
    for off in range(pid, TOTAL, n):
        ...
```

关键实现细节：
- `num_cores` 作为 `tl.constexpr` 传入 kernel，编译器可以优化；
- host 端动态获取，避免硬编码；
- 有兜底值，增强鲁棒性。

## 已知实例

- `exp_ascend_core_num_001` — Hardware Adaptation（收益：规避 10-15% 同步损耗）
  - 动态获取 VEC/CUBE core 数。910B2 默认 VEC=48、CUBE=24；910B3 默认 VEC=64、CUBE=32。按需调整兜底值，避免 forward 中的设备同步。
- `exp_ascend_split_001` — Hardware Adaptation（收益：避免编译失败，恢复访存效率）
  - 910B2/910B3 平台下 BLOCK_SIZE≥65536 会直接编译失败，block 内存超缓存会导致访存效率下降 50% 以上。多次切分可在符合限制的前提下充分利用硬件算力。

## 关联知识

- 相关 runbook: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md)
- 相关 runbook: [matmul core mapping 策略选择（列/行/对角）](../compute-pipeline/core-mapping-strategy-selection.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [移除冗余 compile_hint 恢复编译器优化](../autotuning/traj-general_ascend-tuning_config.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 大 Shape 算子的交错循环负载均衡](../compute-pipeline/traj-general_ascend-load_balance.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](../memory-sram/traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 输出张量创建优化：empty/empty_like 替代 zeros/ones](../memory-sram/traj-general_ascend-memory_init.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend 后端避免用 tl.where 动态计算内存偏移](../meta-skills/traj-general_ascend-syntax_adapt.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Triton-Ascend 切片语法避坑：tl.arange 不是张量](../meta-skills/traj-general_ascend-syntax_pitfall.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Ascend Triton 语法规范避坑聚合](../meta-skills/traj-general_ascend-syntax_spec.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Attention 硬件适配优化聚合](traj-attention-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Ascend 910B 硬件约束：i64 地址、整数比较与 block_ptr boundary_check 标量降级](traj-general_ascend-hardware_constraint.md) — 主题相关：共享 910b2, 910b3, general_ascend
- 相关主题: [Entropy Backward 在 Ascend NPU 上的硬件适配优化](traj-loss-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Matmul 硬件适配优化聚合](traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation

<!-- okf:related:end -->
