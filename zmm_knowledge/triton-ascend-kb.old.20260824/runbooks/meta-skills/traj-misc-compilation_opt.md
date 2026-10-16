---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: misc
dimension: "6.2"
title: 动态序列长度场景使用 do_not_specialize 降低编译开销
description: 针对 speculative decoding 等动态序列长度场景，通过 do_not_specialize 避免为每个长度生成独立 kernel 二进制，将编译时间降低 80% 以上。
tags: [misc, compilation_opt, do_not_specialize, dynamic_shape, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/misc/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# 动态序列长度场景使用 do_not_specialize 降低编译开销

## 概述

在 speculative decoding 等场景中，`max_spec_len` 每次请求的候选 token 数可能不同。如果 Triton 把该参数当作编译期常量（ specialization ），则每个不同长度都会生成一份独立的 kernel 二进制，导致：

- 编译时间随长度分布线性增长；
- 运行时缓存的 kernel 数量膨胀；
- 首次调用新长度时的延迟抖动。

该经验通过在 `@triton.jit` 上显式声明 `do_not_specialize=["max_spec_len"]`，让编译器把该参数视为运行时变量，从而只生成一份 kernel。实测动态序列长度下编译时间降低 80% 以上，执行开销可忽略。

## 共同模式

### 1. 识别动态但非 tile 的关键参数

适合 `do_not_specialize` 的参数特征：

- 运行时变化频繁（如 sequence length、batch size、speculative len）；
- 不决定 block/tile 大小或 constexpr 展开；
- 只参与地址计算或边界 mask。

不适合 `do_not_specialize` 的参数：

- `BLOCK_SIZE` 等决定 tile 大小的 constexpr；
- 影响 `tl.dot` 维度或循环展开次数的参数。

### 2. 直接内联指针算术

kernel 内直接用 `position * (max_spec_len + 1) + num_tokens1` 计算扁平化 2D 地址，无需 host 侧 reshape 或预处理。由于 `max_spec_len` 不参与 tile 计算，运行时求值没有性能损失。

### 3. 最小化寄存器压力

该 kernel 每次调用只处理单个元素，寄存器占用极低，可以保持高 occupancy。这是 `do_not_specialize` 策略的自然延伸：用运行时变量换取编译期灵活性，同时通过极简计算体避免运行时额外开销。

### 4. 只适用于单元素/轻量更新

`bonus_renew` 是单元素插入操作，不适合 batched updates 或大块内存拷贝。对于后者，应考虑其他动态 shape 处理策略（如 padding 到固定 bucket）。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_bonus_renew_001` | 编译时间降低 >80%（动态序列长度） | 对 speculative decoding 的 `max_spec_len` 使用 `do_not_specialize`，避免为每个长度生成独立 kernel，编译时间大幅降低且执行开销可忽略。 |

## 代表 Kernel

以下为 `exp_bonus_renew_001` 对应的 kernel：

```python
@triton.jit(do_not_specialize=["max_spec_len"])
def bonus_renew(
    bonus_token_ids_ptr,
    position,
    output_token_ids_ptr,
    max_spec_len,
    num_tokens1,
):
    bonus_token_id = tl.load(bonus_token_ids_ptr + position)
    tl.store(
        output_token_ids_ptr + position * (max_spec_len + 1) + num_tokens1,
        bonus_token_id,
    )
```

## 已知实例

- `exp_bonus_renew_001` — Compilation Optimization
  - 适用于 speculative decoding 的单元素插入场景；关键优化为 `do_not_specialize=["max_spec_len"]`。
  - 对应 kernel: `raw_experiences/extracted_kernels/166-bonus_renew.py`

## 关联知识

- 开发指南: [Misc 算子优化](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Misc 算子 Scheduling 优化聚合](../compute-pipeline/traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](../memory-sram/pow2-block-constexpr.md) — 主题相关：共享 dynamic_shape
- 相关主题: [Misc 算子内存访问优化聚合](../memory-sram/traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, do_not_specialize
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](../portability/traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, misc

<!-- okf:related:end -->
