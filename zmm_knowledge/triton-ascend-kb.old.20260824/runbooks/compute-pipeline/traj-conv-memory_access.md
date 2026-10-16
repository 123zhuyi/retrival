---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: conv
dimension: "2.4"
title: Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling
description: 针对 Ascend 910B 系列上 Causal Conv1D（Mamba/SSM 核心算子）的内存访问优化经验，核心是通过滑动窗口寄存器复用减少历史 token 重复加载、用 Block Pointer 替代标量 load、以及 Update 阶段的重叠 T-tiling 消除跨 tile 冗余状态读取。
tags: [conv, causal_conv1d, memory_access, sliding_window, block_pointer, tiling, ssm, mamba, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/conv/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling

## 概述

Causal Conv1D 是 Mamba、RWKV 等 State Space Model（SSM）的核心算子，其特点是：

- **小卷积核**（通常 `W=4`），每个输出 token 只依赖当前及前 `W-1` 个输入 token；
- **因果性**：当前 token 不能依赖未来 token；
- **状态递归**：decode/update 阶段需要维护 `conv_states` 缓存历史 token；
- **变长序列**：训练/推理 batch 中序列长度不一致，需要 `cu_seqlens` 或 `chunk_indices` 定位边界。

在 Ascend 910B 上，Causal Conv1D 通常是 **内存带宽受限** 而非计算受限。本组 8 条 experience 的优化目标都是减少全局内存读写次数、提升 load 的向量化程度、以及让历史 token 在寄存器/SRAM 中复用。

**核心 trade-off**：

| 优化手段 | 收益 | 代价/约束 |
|---|---|---|
| **滑动窗口寄存器复用** | Prefill 阶段可减少 75% token load | 卷积核宽度 W 需编译期已知且较小（通常 ≤8） |
| **`tl.make_block_ptr` 替代标量 load** | D≥2048 时可达 2-3x | 要求访问模式规则、stride 已知 |
| **重叠 T-tiling** | Update 阶段节省 15-25% 带宽 | 增加 (W-1) 列的 UB buffer，需处理首 tile 状态拼接 |
| **编译期特化**（`HAS_WEIGHT`/`HAS_BIAS`/`IS_VARLEN`） | 消除运行时分支 | 每个配置组合会生成一个 kernel 变体 |
| **T/D 二维 tiling** | 提升 cache/UB 局部性 | 需调 BT/BD 避免 UB 溢出 |
| **状态更新与计算融合** | 减少一次 kernel launch 和 HBM 往返 | kernel 逻辑更复杂 |

## 共同模式

### 1. 滑动窗口寄存器复用：Prefill 阶段的核心优化

对于 `W=4` 的小卷积核，参考实现会在每个时间步加载 4 个输入 token（当前 + 前 3 个）。但实际上相邻时间步的窗口高度重叠：前一步的 `[t-3, t-2, t-1, t]` 与下一步的 `[t-2, t-1, t, t+1]` 只相差一个新 token。

**优化方案**：用 4 个显式命名寄存器 `col0/col1/col2/col3` 保存当前窗口。每步计算后执行寄存器移位 `col0←col1←col2←col3`，然后只加载 1 个新 token 到 `col3`。

**根因**：
- 全局内存 load 是瓶颈，每个 token 从 HBM 读取一次成本远高于寄存器移位；
- 寄存器移位在编译器层面通过寄存器重命名实现，零周期开销；
- 窗口末尾的 `W-1` 个值恰好就是需要写回 `conv_states` 的状态，状态更新与卷积 load 完全复用。

**收益明显条件**：
- `W` 小且固定（如 4），寄存器占用可接受；
- Prefill 阶段 `seq_len` 较长，load 次数 reduction 被放大；
- 数据布局按 `[T, D]` 或 `[B, T, D]` 连续存储。

**常见坑**：
- 首次迭代需要从 `conv_states` 加载 `W-1` 个历史值初始化窗口；
- 若用数组 `col[W]` 而非命名寄存器，编译器可能无法做寄存器重命名，会生成实际拷贝指令；
- 4 路 FMA 归约应写为单语句 `col0*w0 + col1*w1 + col2*w2 + col3*w3`，让编译器识别为独立乘法 + 加法树。

### 2. Block Pointer 替代标量 load：大 D 场景的必须项

`tl.make_block_ptr` 在 Ascend 上映射为硬件 block transfer（类似 DMA），一条指令搬运整个 tile，并自带边界检查和零填充。而多次独立 `tl.load(weight_ptr + ...)` 需要独立的标量地址计算，无法利用硬件向量化。

**根因**：标量 load 在 D 很大时会成为 MTE2（内存传输引擎）到 Vector 的瓶颈；Block Pointer 可以把整个 `[D_CHK, W]` 权重 tile 或 `[BT, BD]` 输入 tile 一次性搬进 UB。

**实现要点**：
- 输入、权重、输出尽量用 `tl.make_block_ptr`；
- `boundary_check=(0, 1)` + `padding_option="zero"` 处理尾块，避免手写 mask；
- 若需要对权重按 W 方向切片（如 `w0/w1/w2/w3`），采用**混合方案**：先用 `make_block_ptr` 加载整个 `[D_CHK, W]` tile，再用 `tl.extract_slice` 取各列，不要退化为独立标量 load。

### 3. 重叠 T-tiling：Update 阶段消除跨 tile 冗余读取

Update kernel 处理的是 decoding 阶段，输入序列较短但需要对 `conv_states` 做状态更新。当对 T 维度做 tiling 时，相邻 T-tile 的卷积窗口需要共享 `W-1` 个历史 token。

**根因**：若每个 tile 独立从 `conv_states` 读取历史，则除第一个 tile 外，每个 tile 都会重复加载前 `W-1` 个 token。

**优化方案**：
- Tile 0：从 `conv_states` 加载尾部 `W-1` 个历史 token，与当前 tile 输入拼接成 `[D_CHK, (W-1)+T_CHK]` buffer；
- Tile i（i>0）：从输入的 `ti*T_CHK_SIZE - (W-1)` 位置开始加载，前 `W-1` 列自然等于上一 tile 的最后 `W-1` 个 token，无需再次读取 `conv_states`。

**收益**：节省约 `(num_T_tiles - 1) × (W-1) × D_CHK_SIZE` 次全局内存 load。

### 4. T/D 二维 Tiling 与编译期特化

Causal Conv1D 的输入通常是 `[B, T, D]` 或 `[B, D, T]`。二维 tiling 把时间（T）和通道（D）切分成 `(BT, BD)` tile，使工作集适应 UB/SRAM。

**编译期特化**：用 `@triton.heuristics` 把 `HAS_WEIGHT`、`HAS_BIAS`、`USE_INITIAL_STATE`、`USE_FINAL_STATE`、`IS_VARLEN` 等运行时标志转为 `tl.constexpr`：
- 消除主循环内的 `if` 分支；
- 让编译器针对每种配置生成 specialized 代码；
- 注意：每个配置组合会增加编译缓存占用。

### 5. 状态更新与卷积计算融合

Decode/update 阶段通常需要：
1. 从 `conv_states` 读历史；
2. 与当前输入拼接；
3. 做卷积 + bias + activation；
4. 把新的状态写回 `conv_states`。

把这些步骤融合到一个 kernel，避免中间张量写 HBM。

**实现要点**：
- 状态更新只在覆盖状态保存区 `[seq_len - state_len, seq_len)` 的 tile 中执行；
- 用 `tl.insert_slice` 把新输入拼接到历史 buffer；
- 写回时通过 mask 只覆盖需要更新的状态位置。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_causal_conv1d_prefill_002` | 消除 75% token load（整体 1.60x，最高 1.84x） | Prefill 阶段滑动窗口寄存器复用：4 个命名寄存器保存窗口，每步只 load 1 个新 token。 |
| `exp_causal_conv1d_bwd_001` | 35% | Causal Conv1D backward，Block Pointer + 编译期特化 + W 循环展开 + T/D tiling。 |
| `exp_causal_conv1d_001` | 35% | Causal Conv1D forward，Block Pointer + heuristics 特化 + loop 内 load 防 UB 溢出。 |
| `exp_causal_conv1d_update_001` | 35% | Decode 阶段 update，融合状态更新 + 卷积 + SiLU，channel blocking + 向量状态更新。 |
| `exp_causal_conv1d_update_004` | 15-25% 内存带宽 | Update 重叠 T-tiling：相邻 tile 共享 W-1 历史 token，减少 (num_T_tiles-1)×(W-1)×D_CHK_SIZE 次 load。 |
| `exp_causal_conv1d_update_003` | D≥2048 时 2-3x | Block Pointer vs 标量 load 对比；推荐 block_ptr 加载 + extract_slice 切片的混合方案。 |
| `exp_causal_conv1d_states_001` | 25% | `conv_states` 状态更新，2D 向量 IO + 边界 mask + 初始状态累加融合。 |
| `exp_causal_conv1d_varlen_opt_001` | 22% | 变长 Conv1D update，constexpr stride、BLOCK_D=256、`care_padding=False`、条件窗口移位。 |

## 代表 Kernel

选取 `exp_causal_conv1d_prefill_002` 作为代表，原因：

1. **它是本组收益最高的经验**，且收益来源单一清晰——滑动窗口寄存器复用减少 75% token load；
2. 该模式是 Causal Conv1D Prefill 阶段最核心的优化，对 Mamba/SSM 类模型具有普遍参考价值；
3. 代码简洁，能直接展示“命名寄存器 + 移位 + 单语句 FMA 归约”的关键技巧。

```python
# 初始化：从 conv_states 加载历史 + 第一个新 token
col0 = tl.load(conv_states_ptr + 0*stride, mask=mask_d)
col1 = tl.load(conv_states_ptr + 1*stride, mask=mask_d)
col2 = tl.load(conv_states_ptr + 2*stride, mask=mask_d)
col3 = tl.load(x_ptrs, mask=mask_d)                    # 第 1 个新 token

for i in range(seq_len):
    # 4 路 FMA：单语句归约，编译器可并行发射 4 条向量乘法
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3
    if HAS_BIAS:
        acc = acc + bias
    if ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))                 # fp32 SiLU
    tl.store(y_ptrs + i * stride_y, acc, mask=mask_d)

    # 窗口滑动：寄存器移位（编译器重命名，零周期开销）
    col0 = col1
    col1 = col2
    col2 = col3
    # 加载下一个新 token
    col3 = tl.load(x_ptrs + (i+1) * stride_x, mask=mask_d)
```

**关键实现细节**：
- 用 `col0/col1/col2/col3` 4 个显式命名变量，而非数组，确保编译器能做寄存器重命名；
- 卷积归约写成单语句 4 路 FMA，避免循环累积的跨迭代依赖；
- 窗口末尾的 `col1/col2/col3` 就是新的 `conv_states`，无需额外 load/store。

## 已知实例

- `exp_causal_conv1d_prefill_002` — Memory Access （收益 消除 75% token load，整体 1.60x/1.84x）
  - Prefill 阶段滑动窗口寄存器复用，代表 kernel。
- `exp_causal_conv1d_bwd_001` — Memory Access （收益 35%）
  - Causal Conv1D backward，Block Pointer + 编译期特化 + W 循环展开 + T/D tiling。
  - 对应 kernel: `raw_experiences/extracted_kernels/267-causal_conv1d_bwd_kernel.py`
- `exp_causal_conv1d_states_001` — Memory Access （收益 25%）
  - `conv_states` 状态更新，2D 向量 IO + 边界 mask + 初始状态累加融合。
  - 对应 kernel: `raw_experiences/extracted_kernels/268-causal_conv1d_states_fwd_kernel.py`
- `exp_causal_conv1d_varlen_opt_001` — Memory Access （收益 22%）
  - 变长 Conv1D update，constexpr stride、BLOCK_D=256、`care_padding=False`、条件窗口移位。
- `exp_causal_conv1d_update_004` — Memory Access （收益 15-25% 内存带宽）
  - Update 重叠 T-tiling，相邻 tile 共享 W-1 历史 token。
- `exp_causal_conv1d_001` — Memory Access （收益 35%）
  - Causal Conv1D forward，Block Pointer + heuristics 特化 + loop 内 load 防 UB 溢出。
  - 对应 kernel: `raw_experiences/extracted_kernels/266-causal_conv1d_fwd_kernel.py`
- `exp_causal_conv1d_update_001` — Memory Access （收益 35%）
  - Decode 阶段 update，融合状态更新 + 卷积 + SiLU，channel blocking + 向量状态更新。
  - 对应 kernel: `raw_experiences/extracted_kernels/188-_causal_conv1d_update_kernel.py`
- `exp_causal_conv1d_update_003` — Memory Access （收益 D≥2048 时 2-3x）
  - Block Pointer vs 标量 load 分析，推荐 block_ptr 加载 + extract_slice 切片。

## 关联知识

- 开发指南: [Conv 算子优化](../../reference/op-class-convolution.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [卷积算子优化（Causal Conv1D）](../../reference/op-class-convolution.md) — 主题相关：共享 910b2, 910b3, mamba, ssm
- 对应指南: [SSM / 状态空间模型算子优化](../../reference/op-class-ssm.md) — 主题相关：共享 910b2, 910b3, mamba, ssm
- 相关主题: [BlockSparseAttentionGrad (block-sparse backward) on Ascend NPU](../../ops/block-sparse-attention-grad.md) — 同属优化知识，主题相关
- 相关主题: [BlockSparseAttention (block-sparse mask) on Ascend NPU](../../ops/block-sparse-attention.md) — 同属优化知识，主题相关
- 相关主题: [NsaCompressAttentionInfer (NSA compress attention + importance score + topK) on Ascend NPU](../../ops/nsa-compress-attention-infer.md) — 主题相关：共享 sliding_window
- 相关主题: [NsaCompressAttention (NSA compress attention + topk index selection) on Ascend NPU](../../ops/nsa-compress-attention.md) — 主题相关：共享 sliding_window
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 相关主题: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 ssm
- 相关主题: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 同属优化知识，主题相关
- 相关主题: [Flash Attention Causal Skeleton](../../reference/skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, tiling
- 相关主题: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../memory-sram/block-ptr-vs-scalar-load.md) — 同属优化知识，主题相关
- 相关主题: [M>=256 时 BLOCK_M=256（总 block-iter 减半）](../memory-sram/blockm-256-large-m.md) — 同属优化知识，主题相关
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [重叠 T-tiling 保证跨 tile 滑动窗口连续性](../memory-sram/overlapping-t-tiling.md) — 主题相关：共享 causal_conv1d, sliding_window, ssm
- 相关主题: [BLOCK 向上取 2 的幂并声明为 constexpr](../memory-sram/pow2-block-constexpr.md) — 同属优化知识，主题相关
- 相关主题: [寄存器移位复用滑动窗口，消除重复 HBM 读](../memory-sram/register-shift-sliding-window.md) — 主题相关：共享 causal_conv1d, sliding_window, ssm
- 相关主题: [Activation Backward 内存访问优化聚合](../memory-sram/traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../memory-sram/traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, block_pointer, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](../memory-sram/traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](../memory-sram/traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, tiling
- 相关主题: [Misc 算子内存访问优化聚合](../memory-sram/traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](../memory-sram/traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, ssm
- 相关主题: [UB 占用决定 BLOCK_ROWS/tile 大小，必须按算子特征分别自适应](../memory-sram/ub-aware-block-adaptive.md) — 同属优化知识，主题相关
- 相关主题: [1D grid 硬上限 65535，超出用 2D grid 或增大 BLOCK](grid-limit-65535.md) — 同属优化知识，主题相关
- 相关主题: [多核负载均衡分配公式（每 core 处理 block 数差不超过 1）](load-balance-formula.md) — 同属优化知识，主题相关
- 相关主题: [标量循环改写为 tl.dot，提升 Cube 利用率](scalar-loop-to-dot.md) — 主题相关：共享 ssm
- 相关主题: [swizzle2d 块重排提升 L2 复用](swizzle2d-block-reorder.md) — 同属优化知识，主题相关
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Causal Conv1D Prefill/Update 跨 Kernel 优化迁移](traj-conv-cross_kernel.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Causal Conv1D 调度优化：Prefill/Update 布局分野、2D Grid 与 pad_slot 早退](traj-conv-scheduling.md) — 主题相关：共享 910b2, 910b3, causal_conv1d, conv
- 相关主题: [Triton Block Size 通用调优原则](traj-general_perf-block_size.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, block_pointer, tiling
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, ssm
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, mamba, ssm
- 相关主题: [SSM 反向 kernel 标量消除：i64 降级、整数比较与 boundary_check 优化](traj-ssm-scalar_elimination.md) — 主题相关：共享 910b2, 910b3, ssm

<!-- okf:related:end -->
