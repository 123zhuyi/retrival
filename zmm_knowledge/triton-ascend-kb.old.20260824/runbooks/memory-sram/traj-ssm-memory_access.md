---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: ssm
dimension: "1.1"
title: SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵
description: 聚合 2 条 SSM 在 Ascend 910B 上的 memory access 经验，覆盖 KDA intra 反向 kernel 的转置 load 合并化、跨阶段冗余 load 消除、手动指针替代 block_ptr，以及 Gated Delta Rule 前向 kernel 的 128×64 状态矩阵分块与 chunked recurrence，收益 35%。
tags: [ssm, memory_access, transpose_load, register_reuse, manual_pointer, state_matrix_tiling, chunked_recurrence, gated_delta_rule, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/ssm/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵

## 概述

SSM（State Space Model）类 kernel（如 KDA、Gated Delta Rule）的内存访问模式有两个显著特点：
1. **转置子块访问多**：反向传播中经常需要按“转置视角”读取 `dAqk`、`dAkk` 等子块，而原始数据是行优先存储；
2. **状态矩阵大**：隐藏状态 `h` 的 `K×V` 尺寸可能超过单块寄存器容量，需要显式 tiling 并 checkpoint 到 HBM。

本卡 2 条 experience 分别针对这两个特点：
- **KDA intra backward**：把非合并的转置 load 改成“行优先加载 + 寄存器转置”，并消除跨阶段冗余 load；
- **Gated Delta Rule forward**：把 `K×V` 状态矩阵拆成 `128×64` tile，以 chunk 为单位迭代更新，并将中间状态 checkpoint 到 HBM。

**核心 trade-off**：
- 寄存器转置省 HBM 带宽但增加寄存器压力；
- 跨阶段寄存器复用依赖 `tl.debug_barrier()` 不触发 spill，必须实测验证；
- 状态矩阵分块让 SRAM 容纳工作集，但带来更多的 tile 索引和同步逻辑。

## 共同模式

### 1. 转置 load 合并化（行优先 + 寄存器转置）

**问题根因**：需要加载一个转置子块时，若用 `tl.make_block_ptr` 按转置视角设置 `strides=(1, H*BT)`，列方向每次跳跃 `H*BT` 个元素，当 `H*BT` 很大时（如 2048）访存极不连续，MTE2 利用率低。

**收益明显条件**：
- 需要读取转置子块，但原始张量是行优先；
- `H*BT` 较大，导致跨步严重；
- 子块尺寸适中，寄存器转置不会 spill。

**实现要点**：
```python
# 反模式：转置 block_ptr，非合并访存
# p_dAqk = tl.make_block_ptr(dAqk_local, (BT, T_local), (1, H*BT),
#                            (i_i*BC, i_t*BT+i_j*BC), (BC, BC), (0, 1))
# b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))

# 正模式：行优先加载 + 寄存器转置
row_idx = i_t * BT + i_j * BC + o_i[:, None]
col_idx = i_i * BC + o_i[None, :]
dAqk_mask = (row_idx < T_local)[:, None] & (col_idx < BT)[None, :]
p_dAqk_row = dAqk_local + row_idx * H * BT + col_idx
b_dAqk_row = tl.load(p_dAqk_row, mask=dAqk_mask, other=0.0)
b_dAqk = tl.trans(b_dAqk_row)
```
- `tl.trans` 是纯寄存器操作，无额外 HBM 流量；
- mask 仍需覆盖转置后的两个维度。

**常见坑**：
- 若原始数据本身就是列优先，此技巧不适用；
- 子块过大时 `tl.trans` 会增加寄存器压力；
- 指针计算保持 i32，避免 i64 标量降级。

### 2. 跨阶段冗余 load 消除

**问题根因**：多阶段 kernel 中（如 KDA intra 的 Phase 1/1.5/2/2.5），同一数据在不同阶段被重复从 HBM 加载，浪费 MTE2。

**收益明显条件**：
- 多阶段算法，阶段间需要同一块数据；
- 寄存器压力可控，编译器不会在 barrier 后 spill；
- profiling 显示 AIV MTE2 占比较高（如 >30%）。

**实现要点**：
```python
# Phase 1.5：加载一次 b_q
b_q = tl.load(p_q, mask=BC_BK_mask, other=0.0)
# ... 使用 b_q 计算 ...
tl.debug_barrier()
# Phase 2.5：直接使用已加载的 b_q
# ... 继续使用 b_q ...
```
- 数据在 barrier 前后保持寄存器中；
- 需实测验证 `aiv_mte2_time` 是否下降。

**常见坑**：
- 寄存器压力高时 barrier 会触发 spill，复用失效；
- 不要跨 barrier 保留大量数据；
- 该优化收益不确定，必须 profiling 确认。

### 3. 手动指针替代 block_ptr

**问题根因**：`tl.make_block_ptr` 在小 tile 下 boundary_check 会产生标量开销，同时 block_ptr 结构体本身也有一定创建成本。

**收益明显条件**：
- 小 tile（如 BC=16）频繁 load/store；
- `aiv_scalar_ratio` 已确认较高；
- 访问模式不规则或需要复杂 mask。

**实现要点**：
- 用 `tl.arange` 生成偏移，手动计算指针；
- mask 用 fp32 比较生成以降低标量比例；
- 参考通用内存访问优化卡中的 `exp_perf_011`。

**常见坑**：
- 手动指针必须保证 mask 完备，否则越界；
- 大块规则访问仍推荐 block_ptr；
- 指针计算避免 i64。

### 4. 状态矩阵分块（128×64 tiling）

**问题根因**：Gated Delta Rule 的隐藏状态 `h` 形状为 `(K, V)`，当 K、V 较大时无法一次性放入寄存器/SRAM。直接全量加载会导致频繁 spill 或非合并访问。

**收益明显条件**：
- `K >= 128`、`V >= 128`；
- 状态 `h` 在 chunk 迭代中被反复读取和更新；
- 需要把状态 checkpoint 到 HBM 以支持长序列。

**实现要点**：
- 把 `h` 拆成多个 `128×64` tile（如 `b_h1_bv1`、`b_h1_bv2`）；
- 每个 tile 独立参与 `tl.dot` 更新；
- 每个 chunk 结束后把 tile 写回 HBM checkpoint。

**常见坑**：
- 分块后需要维护多个 tile 指针，索引复杂；
- tile 边界需同时覆盖 K 和 V 两个方向；
- 多个 tile 的更新顺序可能影响数值精度。

### 5. Chunked recurrence 与状态 checkpoint

**问题根因**：长序列 SSM 的状态无法全部驻留 SRAM，必须分段计算。每次 chunk 计算需要读取上一 chunk 状态、更新、写回当前 chunk 状态。

**收益明显条件**：
- 序列长度 T 很大，超过 SRAM 容量；
- 状态 `h` 形状固定或按 tile 分块后可管理；
- 需要支持 variable-length（`cu_seqlens`）。

**实现要点**：
- 按 `BT` 分块遍历序列；
- 每个 chunk 内计算 `v_new`、`g` 门控、状态更新；
- chunk 边界把 `h` tile 写入 `h` buffer 作为 checkpoint；
- 支持 `h0` 初始状态和 `ht` 最终状态。

**常见坑**：
- `cu_seqlens` 的 load 必须合并；
- 门控 `g` 通常需要 `safe_exp` 稳定化；
- 最后一个 chunk 的边界需特殊处理。

### 6. 混合精度累加

**问题根因**：状态 `h` 的递归更新对数值误差敏感，但 `tl.dot` 输入用 fp16/bf16 才能发挥 Cube 吞吐。

**收益明显条件**：
- 数据类型为 fp16/bf16；
- 包含递归状态更新或长序列累加；
- 对精度敏感。

**实现要点**：
- 状态 `h` tile 用 fp32 累加；
- `tl.dot` 前把输入 cast 到 fp16/bf16；
- store 前再 downcast。

**常见坑**：
- fp32 tile 占用两倍寄存器，需与 tiling 策略配合；
- 不要在整个递归过程中反复 cast，会增加误差；
- 门控 `g` 的 exp 计算建议用 fp32。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_kda_bwd_intra_memory_opt_001` | 35% | KDA intra 反向：转置 load 改为行优先 + 寄存器转置，消除 Phase 1/1.5 与 Phase 2/2.5 间的冗余 load，手动指针替代 block_ptr；预期 AIV MTE2 从 37.9% 降至 <25%。 |
| `exp_chunk_gated_delta_001` | 35% | Gated Delta Rule 前向：chunked recurrence，隐藏状态 `h` 拆为 128×64 tile，fp32 累加状态，支持 variable-length 序列。 |

## 代表 Kernel

选择 `exp_chunk_gated_delta_001` 作为代表，因为它展示了 **chunked recurrence、状态矩阵分块、混合精度累加、variable-length 支持** 四个 SSM 核心内存访问模式。以下为该 experience 的 kernel 代码：

```python
@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k, v, w, v_new, g, h, h0, ht,
    cu_seqlens, chunk_offsets,
    T,
    H: tl.constexpr, Hg: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    # 状态矩阵拆成两个 128x64 tile，fp32 累加
    b_h1_bv1 = tl.zeros([128, 64], dtype=tl.float32)
    b_h1_bv2 = tl.zeros([128, 64], dtype=tl.float32)

    v_start1 = 0
    v_start2 = 64

    offs_k = tl.arange(0, 128)[:, None]
    offs_v1 = v_start1 + tl.arange(0, 64)[None, :]
    offs_v2 = v_start2 + tl.arange(0, 64)[None, :]
    mask_kv1 = (offs_k < K) & (offs_v1 < V)
    mask_kv2 = (offs_k < K) & (offs_v2 < V)

    # 加载初始状态
    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        ptr_h0_bv1 = h0_ptr + offs_k * V + offs_v1 * 1
        b_h1_bv1 += tl.load(ptr_h0_bv1, mask=mask_kv1, other=0.0).to(tl.float32)
        ptr_h0_bv2 = h0_ptr + offs_k * V + offs_v2 * 1
        b_h1_bv2 += tl.load(ptr_h0_bv2, mask=mask_kv2, other=0.0).to(tl.float32)

    # 主循环：按 chunk 递归
    for i_t in range(NT):
        h_base = h + (boh + i_t) * H * K * V + i_h * K * V

        # checkpoint 当前状态到 HBM
        p_h1_bv1 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_h1_bv1, b_h1_bv1.to(p_h1_bv1.dtype.element_ty), boundary_check=(0, 1))
        p_h1_bv2 = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_h1_bv2, b_h1_bv2.to(p_h1_bv2.dtype.element_ty), boundary_check=(0, 1))

        # 加载门控 g、w、k、v tile
        offs_t_wv = (i_t * BT + tl.arange(0, BT))[:, None]
        offs_k_wv = tl.arange(0, 128)[None, :]
        mask_w = (offs_t_wv < T) & (offs_k_wv < K)
        w_base = w + bos * H * K + i_h * K
        ptr_w = w_base + offs_t_wv * stride_w + offs_k_wv * 1
        b_w = tl.load(ptr_w, mask=mask_w, other=0.0)

        k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
        p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + bos + i_h * T_max + last_idx)

        offs_t = i_t * BT + tl.arange(0, BT)
        mask_t = offs_t < T
        g_ptr = g + bos + i_h * T_max
        b_g = tl.load(g_ptr + offs_t, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        # 更新 V 的第一段 tile
        offs_t_v = (i_t * BT + tl.arange(0, BT))[:, None]
        mask_v1 = (offs_t_v < T) & (offs_v1 < V)
        v_base = v + bos * H * V + i_h * V
        ptr_v1 = v_base + offs_t_v * stride_v + offs_v1 * 1
        b_v1 = tl.load(ptr_v1, mask=mask_v1, other=0.0)
        b_v_new1 = b_v1.to(tl.float32)
        b_v_new1 -= tl.dot(b_w, b_h1_bv1.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new1 = tl.make_block_ptr(v_new + bos * H * V + i_h * V,
                                         (T, V), (stride_v, 1),
                                         (i_t * BT, v_start1), (BT, 64), (1, 0))
            tl.store(p_v_new1, b_v_new1.to(p_v_new1.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new1 = b_v_new1 * b_g[:, None]
            b_h1_bv1 = b_h1_bv1 * b_g_last

        b_v_new1 = b_v_new1.to(k.dtype.element_ty)
        b_h1_bv1 += tl.dot(b_k, b_v_new1)

        # 更新 V 的第二段 tile
        mask_v2 = (offs_t_v < T) & (offs_v2 < V)
        ptr_v2 = v_base + offs_t_v * stride_v + offs_v2 * 1
        b_v2 = tl.load(ptr_v2, mask=mask_v2, other=0.0)
        b_v_new2 = b_v2.to(tl.float32)
        b_v_new2 -= tl.dot(b_w, b_h1_bv2.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            p_v_new2 = tl.make_block_ptr(v_new + bos * H * V + i_h * V,
                                         (T, V), (stride_v, 1),
                                         (i_t * BT, v_start2), (BT, 64), (1, 0))
            tl.store(p_v_new2, b_v_new2.to(p_v_new2.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new2 = b_v_new2 * b_g[:, None]
            b_h1_bv2 = b_h1_bv2 * b_g_last

        b_v_new2 = b_v_new2.to(k.dtype.element_ty)
        b_h1_bv2 += tl.dot(b_k, b_v_new2)

    # 写最终状态
    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht1_bv1 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start1), (128, 64), (1, 0))
        tl.store(p_ht1_bv1, b_h1_bv1.to(p_ht1_bv1.dtype.element_ty), boundary_check=(0, 1))
        p_ht1_bv2 = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start2), (128, 64), (1, 0))
        tl.store(p_ht1_bv2, b_h1_bv2.to(p_ht1_bv2.dtype.element_ty), boundary_check=(0, 1))
```

**关键实现细节**：
- 隐藏状态 `h` 显式拆成 `128×64` 两个 fp32 tile，匹配 Ascend 寄存器容量；
- 每个 chunk 结束后把状态 checkpoint 到 `h` buffer，支持长序列和 variable-length；
- `k` 用 `block_ptr` 加载，`v` 用手动指针加载，根据访问模式选择最优方式；
- `b_g = safe_exp(b_g_last - b_g)` 保证门控指数稳定；
- `tl.dot` 输入 cast 到 fp16/bf16，状态累加保持 fp32。

## 已知实例

- `exp_kda_bwd_intra_memory_opt_001` — Memory Access（收益 35%）
  - KDA intra 反向：转置 load 合并化、跨阶段冗余 load 消除、手动指针替代 block_ptr；预期 AIV MTE2 从 37.9% 降至 <25%。
- `exp_chunk_gated_delta_001` — Memory Access（收益 35%）
  - Gated Delta Rule 前向：chunked recurrence、128×64 状态矩阵分块、fp32 状态累加、variable-length 支持，代表 kernel 见上。

## 关联知识

- 开发指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 910b2, 910b3, ssm
- 对应指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 gated_delta_rule, ssm
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](../compute-pipeline/load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [per-channel 参数标量 load 路径，避免向量化 int 除法降级](../compute-pipeline/scalar-load-channel-index.md) — 同属优化知识，主题相关
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](../compute-pipeline/scalar-loop-to-dot.md) — 主题相关：共享 ssm
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, ssm
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [SSM Compute 优化聚合](../compute-pipeline/traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化](../compute-pipeline/traj-ssm-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [Work-Stealing 动态负载均衡（不均匀任务分发）](../compute-pipeline/work-stealing-dynamic-load-balance.md) — 同属优化知识，主题相关
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](overlapping-t-tiling.md) — 主题相关：共享 ssm
- 相关主题: [pertensor scale 必须物化为连续 [1,N]，禁止 stride-0 广播 load](pertensor-scale-materialize.md) — 同属优化知识，主题相关
- 相关主题: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](preload-extract-slice-eliminate-reread.md) — 主题相关：共享 register_reuse
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](register-shift-sliding-window.md) — 主题相关：共享 ssm
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, manual_pointer, memory_access, register_reuse
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, register_reuse

<!-- okf:related:end -->
