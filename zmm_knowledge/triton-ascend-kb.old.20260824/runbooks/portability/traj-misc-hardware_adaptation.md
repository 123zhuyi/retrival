---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: misc
dimension: "4.1"
title: SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡
description: 聚合 3 条 misc hardware adaptation 经验，覆盖 SDA 前向与反向在 Ascend 910B 上的结构化 tiling、HiVM Cube/Vector 自动均衡、多缓冲预取与 CANN 后端扩展标志，收益 35%–45%。
tags: [misc, hardware_adaptation, sda, scaled_dot_attention, ascend_npu, hivm, multibuffer, cube_vector_balance, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/misc/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡

## 概述

SDA（Scaled Dot Attention）是 Attention 的一类紧凑实现，常见于特定模型或训练框架。把它直接迁移到 Ascend 910B 时，会面临：
1. **Cube/Vector 单元不均衡**：matmul（Cube）与 mask/softmax/elementwise（Vector）交替执行，调度不当会导致某类单元空等；
2. **KV 块复用不足**：dK/dV 反向需要反复加载 Q、do、d 等张量，MTE2 压力大；
3. **后端扩展能力差异**：Ascend/CANN 提供了 `enable_hivm_auto_cv_balance`、`multibuffer` 等后端标志，需要显式启用才能发挥硬件能力。

本卡 3 条 experience 分别覆盖：
- `exp_sda_fwd_ascend_001`（35%）：SDA 前向 Ascend 路径；
- `exp_sda_bwd_q_001`（35%）：SDA 反向 dQ；
- `exp_sda_bwd_kv_ascend_001`（45%）：SDA 反向 dK/dV，收益最高。

**核心 trade-off**：
- 更大的 `BLOCK_R/BLOCK_C` 提升 Cube 利用率，但增加 UB/L0 压力；
- 多缓冲预取隐藏 MTE2 延迟，但消耗额外 workspace；
- Cube/Vector 自动均衡依赖后端启发式，极端 shape 下可能需要手动调整 `tile_mix_*`。

## 共同模式

### 1. 结构化 Tiling：BLOCK_R × BLOCK_C

**问题根因**：Attention 类 kernel 的内存访问量随序列长度平方增长。没有合理分块时，中间结果（如 attention score）无法驻留寄存器/L0，导致频繁 HBM 往返。

**收益明显条件**：
- 序列长度 S 较大（≥ 1K）；
- head_dim H 固定且较小（如 64/128）；
- 需要同时优化前向和反向。

**实现要点**：
```python
SEQ_LEN = S // 2
num_c = tl.cdiv(SEQ_LEN, BLOCK_C)
group_size = N // num_group
for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
    idx_b = task_id // (num_group * num_c)
    idx_group = task_id // num_c % num_group
    idx_c = task_id % num_c
    ...
```
- 沿序列维度切分为 `BLOCK_R`（query 方向）和 `BLOCK_C`（key/value 方向）；
- 每个 task 负责一个 KV block，再内层遍历 `group_size` 个 query block；
- `BLOCK_C` 通常较小（64）以让 KV block 驻留，`BLOCK_R` 较大（256）以提升 Cube 吞吐。

**常见坑**：
- `BLOCK_R * BLOCK_C` 超过 L0C 容量导致 spill；
- `group_size` 不是 `N` 的约数时最后一个 group 需要 mask；
- 反向时 Q/K/V 的 stride 容易混淆。

### 2. Cube/Vector 自动均衡：HiVM 后端标志

**问题根因**：Ascend NPU 上 Cube（矩阵乘）和 Vector（逐元素/归约）是异构单元。Attention 中 softmax、mask、scale 是 Vector，而 QK^T、PV 是 Cube，若 loop 划分不当，一方会等待另一方。

**收益明显条件**：
- 使用 Ascend 910B2/B3；
- kernel 内 Cube 与 Vector 操作交错；
- 观察到 AIV/AIC 占用率不均衡。

**实现要点**：
```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_R": 256,
                "BLOCK_C": 64,
                "multibuffer": True,
                "unit_flag": True,
                "set_workspace_multibuffer": 2,
                "enable_hivm_auto_cv_balance": True,
                "tile_mix_vector_loop": 2,
                "tile_mix_cube_loop": 2,
            }
        )
    ],
    key=["N", "S", "H"],
)
```
- `enable_hivm_auto_cv_balance=True` 让后端自动拆分 Cube/Vector 循环；
- `tile_mix_vector_loop` / `tile_mix_cube_loop` 控制混合粒度；
- `unit_flag` 启用更细粒度的单元调度。

**常见坑**：
- 这些标志是 CANN/HiVM 扩展，在非 Ascend 后端会报错；
- 自动均衡并非万能，极端小 head_dim 或大 S 时仍需手动调参；
- 不同 CANN 版本支持的 flag 集合可能不同。

### 3. 多缓冲预取：multibuffer 与 workspace

**问题根因**：反向 dK/dV 在内层循环中需要多次读取 Q、do、d 等张量。每次循环从 HBM 重新加载会受 MTE2 带宽限制。

**收益明显条件**：
- 内层循环次数多（`group_size` 大）；
- 每次循环读取的数据量固定且可预测；
- 后端支持 double/triple buffering。

**实现要点**：
```python
@triton.Config(
    {
        "multibuffer": True,
        "set_workspace_multibuffer": 2,
    }
)
```
- `multibuffer=True` 启用多缓冲；
- `set_workspace_multibuffer=2` 指定缓冲深度；
- 保证循环内 load/compute 模式固定，便于后端做软件流水线。

**常见坑**：
- workspace 大小设置错误会导致运行时内存错误；
- 多缓冲只在循环内存在独立 load 时有效，若数据依赖则无法预取；
- 缓冲深度过大可能挤占 L0/UB。

### 4. 分组任务分解：B × num_group × num_c

**问题根因**：dK/dV 的每个输出元素需要聚合多个 query 位置的贡献。按 `(batch, group, kv_block)` 三维分解任务，可以让不同 program 并行计算不同 KV block 的梯度。

**收益明显条件**：
- 多头注意力，head 数 N 可被 `num_group` 整除；
- KV block 之间无写冲突（dK/dV 每个位置只由对应 KV block 贡献）；
- grid 并行度足够。

**实现要点**：
```python
for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
    idx_b = task_id // (num_group * num_c)
    idx_group = task_id // num_c % num_group
    idx_c = task_id % num_c
    idx_n = idx_group * group_size + idx_ingroup  # 内层循环
```
- 外层并行覆盖 batch、group、KV block；
- 内层循环遍历 group 内的 head；
- 每个 task 累加自己的 dK/dV block，最后 store。

**常见坑**：
- `task_id` 到多维索引的除法在 kernel 内会标量降级，尽量保证相关维度是 `tl.constexpr`；
- 内层循环变量如果是运行时值，注意类型转换；
- group 与 head 的映射要符合 PyTorch 的 weight layout。

### 5. 反向 mask：块内因果/局部 mask

**问题根因**：反向 dK/dV 也需要考虑因果或局部 attention mask。在分块实现中，常用一个布尔/浮点 mask 来屏蔽非法位置。

**收益明显条件**：
- 使用 causal/local attention；
- 反向需要与正向一致的 mask；
- mask 可以按 block 预计算，避免内层重复生成。

**实现要点**：
```python
offs_r_local = tl.arange(0, BLOCK_C)[:, None]
offs_c_local = tl.arange(0, BLOCK_C)[None, :]
chunk_idx_r = offs_r_local // BLOCK_SIZE
chunk_idx_c = offs_c_local // BLOCK_SIZE
block_mask_bool = chunk_idx_r == chunk_idx_c
block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6
```
- 先构造 block 级别的 mask；
- 传给 micro kernel 参与 softmax/gradient 计算；
- 用 `-1e6` 浮点 mask 兼容 fp32/bf16。

**常见坑**：
- mask 的 dtype 与计算 dtype 不匹配导致数值异常；
- causal 方向（上三角/下三角）在反向时要与正向相反；
- `BLOCK_SIZE` 必须与 causal 窗口对齐。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_sda_fwd_ascend_001` | 35% | SDA 前向 Ascend 910B 硬件适配：结构化 tiling + HiVM Cube/Vector 均衡 + 多缓冲预取，提升前向吞吐。 |
| `exp_sda_bwd_q_001` | 35% | SDA 反向 dQ Ascend 适配：利用 NPU 专用 intrinsics 标志优化 dQ 计算的内存与单元利用率。 |
| `exp_sda_bwd_kv_ascend_001` | 45% | SDA 反向 dK/dV Ascend 适配：分组任务分解 + CANN 后端扩展标志 + multibuffer，减少 MTE2 与单元空等。 |

## 代表 Kernel

选择 `exp_sda_bwd_kv_ascend_001` 作为代表，因为它收益最高（45%），且完整展示了 **结构化 tiling、分组任务分解、HiVM Cube/Vector 均衡、多缓冲预取、反向 block mask** 五个硬件适配核心模式。

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_R": 256,
                "BLOCK_C": 64,
                "multibuffer": True,
                "unit_flag": True,
                "set_workspace_multibuffer": 2,
                "enable_hivm_auto_cv_balance": True,
                "tile_mix_vector_loop": 2,
                "tile_mix_cube_loop": 2,
            }
        )
    ],
    key=["N", "S", "H"],
)
@triton.jit
def kernel_sda_bwd_kv_left(
    q,
    k,
    v,
    do,
    d,
    dk,
    dv,
    lse,
    scale: tl.constexpr,
    num_group: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    STRIDE_Q_B: tl.constexpr,
    STRIDE_Q_N: tl.constexpr,
    STRIDE_Q_S: tl.constexpr,
    STRIDE_Q_H: tl.constexpr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_N: tl.constexpr,
    STRIDE_K_S: tl.constexpr,
    STRIDE_K_H: tl.constexpr,
    STRIDE_V_B: tl.constexpr,
    STRIDE_V_N: tl.constexpr,
    STRIDE_V_S: tl.constexpr,
    STRIDE_V_H: tl.constexpr,
    STRIDE_D_B: tl.constexpr,
    STRIDE_D_N: tl.constexpr,
    STRIDE_D_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOW_TYPE: tl.constexpr = tl.bfloat16,
    HIGH_TYPE: tl.constexpr = tl.float32,
):
    pid = tl.program_id(axis=0)
    SEQ_LEN = S // 2
    num_c = tl.cdiv(SEQ_LEN, BLOCK_C)
    group_size = N // num_group

    # 三维任务分解：batch × group × kv_block
    for task_id in range(pid, B * num_group * num_c, tl.num_programs(axis=0)):
        idx_b = task_id // (num_group * num_c)
        idx_group = task_id // num_c % num_group
        idx_c = task_id % num_c
        idx_h = tl.arange(0, H)

        # K/V block 指针（当前 task 负责）
        ptr_k = (
            k
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_v = (
            v
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )
        ptr_dk = (
            dk
            + idx_b * STRIDE_K_B
            + idx_group * STRIDE_K_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_K_S
            + idx_h[None, :] * STRIDE_K_H
        )
        ptr_dv = (
            dv
            + idx_b * STRIDE_V_B
            + idx_group * STRIDE_V_N
            + (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] * STRIDE_V_S
            + idx_h[None, :] * STRIDE_V_H
        )

        mask_kv = (idx_c * BLOCK_C + tl.arange(0, BLOCK_C))[:, None] < S
        block_k = tl.load(ptr_k, mask=mask_kv, other=0.0)
        block_v = tl.load(ptr_v, mask=mask_kv, other=0.0)
        block_dk = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)
        block_dv = tl.full([BLOCK_C, H], 0.0, dtype=HIGH_TYPE)

        # 内层遍历 group 内所有 head/query 块，累加 dK/dV
        for idx_ingroup in range(group_size):
            idx_n = idx_group * group_size + idx_ingroup

            # block-level mask（因果/局部 attention）
            offs_r_local = tl.arange(0, BLOCK_C)[:, None]
            offs_c_local = tl.arange(0, BLOCK_C)[None, :]
            chunk_idx_r = offs_r_local // BLOCK_SIZE
            chunk_idx_c = offs_c_local // BLOCK_SIZE
            block_mask_bool = chunk_idx_r == chunk_idx_c
            block_mask = (block_mask_bool.to(HIGH_TYPE) - 1.0) * 1e6

            block_dk, block_dv = micro_kernel_bwd_kv(
                q,
                block_k,
                block_v,
                do,
                d,
                block_dk,
                block_dv,
                lse,
                scale,
                idx_c * BLOCK_C,
                block_mask,
                idx_b,
                idx_n,
                idx_h,
                S,
                STRIDE_Q_B,
                STRIDE_Q_N,
                STRIDE_Q_S,
                STRIDE_Q_H,
                STRIDE_D_B,
                STRIDE_D_N,
                STRIDE_D_S,
                BLOCK_C,
                LOW_TYPE,
                HIGH_TYPE,
            )

        tl.store(ptr_dk, block_dk.to(LOW_TYPE), mask=mask_kv)
        tl.store(ptr_dv, block_dv.to(LOW_TYPE), mask=mask_kv)
```

**关键实现细节**：
- autotune 配置里启用 `enable_hivm_auto_cv_balance`、`multibuffer`、`unit_flag` 等 Ascend 专用标志；
- 外层任务分解为 `(B, num_group, num_c)`，每个 task 负责一个 KV block；
- 内层按 `group_size` 遍历 head，累加 dK/dV；
- K/V block 只加载一次，Q/do/d 在内层 micro kernel 中按需读取；
- block-level mask 在 micro kernel 外部预计算，避免重复生成；
- dK/dV 累加使用 `HIGH_TYPE`（fp32），最后 cast 回 `LOW_TYPE` 写回。

## 已知实例

- `exp_sda_fwd_ascend_001` — Hardware Adaptation（收益 35%）
  - SDA 前向 Ascend 910B 硬件适配：结构化 tiling + HiVM Cube/Vector 均衡 + 多缓冲预取。对应 kernel: `raw_experiences/extracted_kernels/259-kernel_sda_fwd_up.py`。
- `exp_sda_bwd_q_001` — Hardware Adaptation（收益 35%）
  - SDA 反向 dQ Ascend 适配：NPU 专用 intrinsics 标志优化 dQ 计算。对应 kernel: `raw_experiences/extracted_kernels/262-kernel_sda_bwd_q_up.py`。
- `exp_sda_bwd_kv_ascend_001` — Hardware Adaptation（收益 45%）
  - SDA 反向 dK/dV Ascend 适配：分组任务分解 + CANN 后端扩展标志 + multibuffer。对应 kernel: `raw_experiences/extracted_kernels/264-kernel_sda_bwd_kv_left.py`。

## 关联知识

- 开发指南: [融合算子综合页](../../reference/op-class-fused_ops.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [GEMM 操作数保持 low_dtype，fp32 累加](../algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 同属优化知识，主题相关
- 相关主题: [multibuffer/unit_flag 等编译选项需实测验证，禁止默认开启](../compute-pipeline/multibuffer-empirical.md) — 主题相关：共享 multibuffer
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 hivm
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](../compute-pipeline/scalar-loop-to-dot.md) — 同属优化知识，主题相关
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Misc 算子 Scheduling 优化聚合](../compute-pipeline/traj-misc-scheduling.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Misc 算子内存访问优化聚合](../memory-sram/traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [tl.dot vs torch.matmul 精度差异诊断](../meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [动态序列长度场景使用 do_not_specialize 降低编译开销](../meta-skills/traj-misc-compilation_opt.md) — 主题相关：共享 910b2, 910b3, misc
- 相关主题: [Attention 硬件适配优化聚合](traj-attention-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation, hivm
- 相关主题: [Ascend 硬件资源适配规则聚合](traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Entropy Backward 在 Ascend NPU 上的硬件适配优化](traj-loss-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend_npu, hardware_adaptation
- 相关主题: [Matmul 硬件适配优化聚合](traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend_npu, hardware_adaptation

<!-- okf:related:end -->
