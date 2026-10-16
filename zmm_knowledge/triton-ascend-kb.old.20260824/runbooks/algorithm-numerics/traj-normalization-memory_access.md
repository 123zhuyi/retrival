---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: normalization
dimension: "3.3"
title: Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion
description: 针对 Ascend 910B 系列上 LayerNorm、RMSNorm、GroupNorm 及其融合变体的内存访问优化经验，核心是按 hidden size 选择 tile 策略、Two-Pass 降低寄存器压力、Kernel Fusion 减少中间结果写回 HBM。
tags: [normalization, layernorm, rmsnorm, groupnorm, memory_access, tiling, two_pass, kernel_fusion, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/normalization/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-11T00:00:00Z"
---

# Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion

## 概述

Normalization 算子（LayerNorm、RMSNorm、GroupNorm）是典型的 **内存带宽 + 规约** 混合瓶颈算子。其特点是：每个样本/通道独立做规约，计算量小、访存量大，且常常需要与 Add、Bias、Activation、Quantization、RoPE 等操作相邻出现。

本组 24 条 experience 覆盖 Ascend 910B2/910B3 上 Normalization 前向与反向的内存访问优化，主要解决三类问题：

1. **大 hidden size 导致 UB/寄存器溢出**：当 `n_cols ≥ 4096` 时，单次加载整行会超出 NPU Unified Buffer 容量，需要列方向 tiling。
2. **多算子串行带来重复 HBM 读写**：Add → Norm → Bias → Quant / RoPE 等链式操作如果拆成多个 kernel，中间张量反复写读 HBM。
3. **反向传播 dW/dB 的原子竞争**：多个 program 同时写同一 weight gradient 地址，原子 add 成为瓶颈。

**核心 trade-off**：

| 优化手段 | 收益 | 代价/约束 |
|---|---|---|
| **小 hidden size 整行向量化**（n_cols < 2048） | 减少 loop/边界开销，提升带宽利用率 | 仅适用于小维度；大维度会 UB 溢出 |
| **大 hidden size 列分块**（n_cols ≥ 2048） | 避免 UB 溢出，支持 4096+ 大模型 | 引入 Two-Pass，同一样本数据需加载两次 |
| **Two-Pass 算法** | 降低寄存器压力，防止 spill | 增加内存带宽（重载 X），小维度不必要 |
| **Kernel Fusion**（Add+Norm+Bias+Quant/RoPE） | 消除中间张量 HBM 往返 | kernel 逻辑复杂，寄存器/UB 预算紧张 |
| **Grid-stride / Persistent Kernel** | 固定 grid 处理任意 batch/seq，提升 occupancy | 单 program 负责多行，需保证负载均衡 |
| **dW 部分和 + 二次规约** | 消除原子竞争 | 需要额外 workspace 和 reduction kernel |
| **FP32 规约 + 低精度存储** | 数值稳定 | 增加少量寄存器/转换开销 |

**适用条件**：fp16/bf16/fp32，n_cols ≥ 128，batch/seq 长度不限；小维度（<2048）优先 no-tiling，大维度（≥2048）优先 column tiling + Two-Pass。

## 共同模式

### 1. 按 hidden size 选择 tile 策略：小整行、大分块

Normalization 的列方向就是需要规约的维度。Ascend NPU 的 UB 容量有限，不能一律整行加载。

**根因**：RMSNorm/LayerNorm 需要先算整行的统计量（mean、variance），再逐元素缩放。若整行超过 UB，必须分块；若整行很小，分块反而带来 loop 和 boundary 开销。

**决策边界**（本组经验共识）：
- **n_cols < 2048**：用 no-tiling，一次性 `tl.arange(0, BLOCK_SIZE)` 加载整行，最大化向量化带宽。
- **2048 ≤ n_cols < 4096**：开始需要 column tiling，块大小通常取 1024/2048。
- **n_cols ≥ 4096**：必须 Two-Pass + column tiling，先分块算 `sum_squares`/`mean`，再分块做归一化。

**实现要点**：
- `BLOCK_SIZE` 要对齐硬件向量宽度（128 bit / 256 bit，即 16/32 个 fp16 元素），保证合并访问。
- 大维度 tiling 时，第一遍规约结果（`rstd`、`mean`）通常保留在标量/短向量寄存器中，供第二遍复用，而不是写回 HBM。
- 对尾块使用 `mask = cols < hidden_size` + `other=0.0`，避免越界。

### 2. Two-Pass 算法：用带宽换寄存器空间

Two-Pass 把计算拆成“统计量规约”和“归一化应用”两个阶段。第一阶段遍历列求 `sum`、`sum_squares`；第二阶段重载 X，用统计量做缩放和 affine。

**根因**：单 Pass 需要同时保留输入 X、中间统计量、输出空间，寄存器压力大；Two-Pass 每阶段只处理一类数据，可显著降低 live variable 数量，避免 register spill。

**收益明显条件**：
- 大 hidden size（≥2048），单 Pass 寄存器不够用；
- batch/seq 较大，重载 X 的额外带宽可以被足够的并行度隐藏。

**常见坑**：
- 小维度（<2048）不要用 Two-Pass，重载 X 的带宽损失会超过寄存器收益。
- 两阶段之间若把中间结果 S（如 Add+Norm 中的 X+R）写 HBM 再读回，会抵消 Two-Pass 的寄存器收益；应把 S 保留在共享内存或寄存器中。
- LayerNorm backward 的 Two-Pass 更复杂：第一遍算 `ds_dx_hat`、`ds`，第二遍算 `dX`，需仔细管理中间统计量。

### 3. Grid-stride / Persistent Kernel：固定 grid 处理任意行数

使用 `for i in tl.range(pid, n_rows, num_progs)` 或 `for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES)`，让一个 program 顺序处理多行。

**根因**：Normalization 每行计算独立，行维度天然可并行。若 grid 大小与行数 1:1，行数少时 occupancy 低、行数多时 launch 开销大。Grid-stride 把 grid 固定到 core 数附近，提升利用率并减少 launch 开销。

**收益明显条件**：
- batch/seq 长度变化大；
- 大 batch（如 LLM prefill 阶段），需要把 core 跑满。

**实现要点**：
- `num_progs` 或 `NUM_CORES` 取硬件核心数，grid 大小与之匹配。
- 循环内使用 `tl.range(..., num_stages=N)` 启用软件流水，隐藏下一行加载延迟。
- Weight/Bias 在行间共享，应提到循环外加载一次，避免每行重复读取。

### 4. Kernel Fusion：把 Add、Norm、Bias、Quant、RoPE 合并

本组大量 experience 将相邻算子融合：
- `Add + RMSNorm`
- `Add + LayerNorm + Bias`
- `RMSNorm + RoPE`（partial RoPE）
- `Split QK + RMSNorm + Copy`
- `Add + RMSNorm + Bias + Quantization`

**根因**：每个独立 kernel 都意味着中间张量写 HBM 再读回。Normalization 本身计算量小，HBM 往返占比高。融合后中间结果驻留寄存器/SRAM，带宽大幅下降。

**收益明显条件**：
- 下游操作紧接 Normalization（如 Add+Norm 后接 MLP/Attention）；
- 量化、RoPE 等操作不需要全局可见的中间结果；
- hidden size 不会导致融合后 UB 溢出。

**与其他优化的关系**：
- 融合 **补充** Two-Pass：Two-Pass 若把 S 写 HBM 会损失性能，融合要求 S 必须留在寄存器/共享内存。
- 融合 **受限于** Tiling：融合后 kernel 占用的 UB/寄存器更多，大 hidden size 需要更保守的 tile 大小。
- 量化场景下，融合可同时完成 `cast(tl.int8, overflow_mode="saturate")`，避免单独的量化 kernel。

**常见坑**：
- 不要为了融合而融合：若融合导致 register spill，性能可能反而下降。
- `tl.extract_slice` 用于量化分块时，注意 slice 大小与 `COL_BLOCK_SIZE` 对齐。
- Partial RoPE 融合时，需区分 ROPE/NOPE 子空间，只对 ROPE 维度做 sin/cos 旋转。

### 5. FP32 规约 + 低精度存储

所有 Normalization 经验都强调：mean、variance、sum of squares、gradient 等规约量用 FP32 计算，最终输入/输出保持 fp16/bf16。

**根因**：平方和、指数、倒数等操作在 fp16/bf16 下容易溢出/下溢，导致训练发散或推理精度损失。FP32 规约只需少量额外寄存器，但大幅提升数值稳定性。

**实现要点**：
- 加载输入后立即 `.to(tl.float32)`，规约完成后再 cast 回输出类型。
- backward 中的 `dW`、`dB` 累加必须用 FP32，否则大批量下会累积显著误差。
- LLAMA/GEMMA 等 casting mode 要区分：LLAMA 通常在 rstd 计算后保持 FP32，GEMMA 可能有不同的量化/缩放约定。

### 6. dW/dB 部分和：避免原子竞争

反向传播中，多个 program 需要把各自算出的 weight gradient 累加到同一块全局内存。`tl.atomic_add` 在竞争严重时成为瓶颈。

**根因**：atomic add 会序列化同一地址的写入，尤其是在 dW 维度小（如 hidden_size 不大）时，多个 program 命中同一列的概率高。

**解决方案**：
- 每个 program 把 dW 累加到自己的 workspace slice（`dW_ptr + pid * n_cols`）。
- 最后用一个轻量 reduction kernel 或 host 端把 `num_programs × n_cols` 的部分和规约成 `n_cols`。
- 当 `n_rows` 很大、竞争可被稀释时，可直接用 `tl.atomic_add` 换取省去二次规约。

### 7. Cache Policy：让一次性数据不污染缓存

Normalization 中输入 X 通常只读一次（Two-Pass 除外），而 Weight 被多行复用。合理设置 cache modifier 可以提升有效缓存命中率。

**经验组合**：
- 对 X、R、grad_y 等一次性数据使用 `eviction_policy="evict_first"`，避免占满缓存。
- 对需要二次加载的 S（融合 Add+Norm 的中间和）使用 `.ca`（cache all）或 `.cg` 根据是否需要复用选择。
- Weight/Bias 提到循环外加载，靠寄存器/缓存复用。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_rmsnorm_fused_quant_001` | 50% | Add + RMSNorm + Bias + INT8 Quantization 四合一融合，权重提前加载、量化分块、`tl.extract_slice` 减少重载。 |
| `exp_rmsnorm_grad_dx_001` | 40% | RMSNorm backward 中 dX 计算，single-pass tiling 把规约和梯度计算合并到一次 UB 内完成，减少约 50% X 读取。 |
| `exp_gn_backward_001` | 35% | GroupNorm backward，空间维度 tiling + 两阶段局部累加，避免原子操作和全局同步。 |
| `exp_rmsnorm_npu_001` | 35% | 大 hidden size 的 Fused Add RMSNorm，column blocking 防 UB 溢出，Two-Pass + `evict_first`/`.cg` cache 策略。 |
| `exp_rmsnorm_bwd_large_cols_001` | 35% | 大列 RMSNorm backward，Two-Pass + 2D tiling，dW 可选 deterministic 部分和或 atomic add。 |
| `exp_rmsnorm_grad_dw_001` | 35% | dW 计算，2D tiling + per-program 部分和，避免 atomic 竞争，支持非连续 stride。 |
| `exp_fused_rms_norm_001` | 35% | Add + RMSNorm 融合，residual 原地写回，FP32 variance，Two-Pass 加载在大 N 下摊平。 |
| `exp_layernorm_bwd_fused_001` | 35% | Fused Add + LayerNorm backward，动态 `BLOCK_SIZE_M`、循环内 atomic dW/dB、融合 residual gradient。 |
| `exp_fused_split_qk_norm_001` | 35% | Split QK + 双 RMSNorm + Copy 融合，输入单遍加载，顺序访问 Q/K-nope/K-pe 三段。 |
| `exp_rmsnorm_bwd_fused_001` | 25% | Fused Add + RMSNorm backward，register 级 dW 累加，LLAMA/GEMMA casting mode。 |
| `exp_rmsnorm_backward_001` | 25% | 小列 RMSNorm backward（n_cols < 2048），整行向量化 + per-pid dW 部分和 + LLAMA/GEMMA casting。 |
| `exp_layernorm_bwd_001` | 25% | 大列 LayerNorm backward（n_cols ≥ 4096），自适应 `BLOCK_SIZE_M`、FP32 规约、atomic dW/dB。 |
| `exp_rmsnorm_tiled_001` | 25% | 大列 RMSNorm 前向（n_cols ≥ 2048），column tiling + Two-Pass + grid-stride + cache policy。 |
| `exp_rmsnorm_backward_tiled_001` | 25% | 大列 RMSNorm backward（n_cols ≥ 2048），column tiling + Two-Pass + per-pid dW 部分和 + 软件流水。 |
| `exp_rmsnorm_fused_001` | 25% | 小列 Fused Add RMSNorm（n_cols < 2048），2D vector loading + 权重预加载 + single-pass 融合。 |
| `exp_rmsnorm_infer_001` | 25% | RMSNorm 推理，grid-stride + Two-Pass + FP32 累加 + tiled coalesced memory access。 |
| `exp_rmsnorm_001` | 15% | 通用 RMSNorm，grid-stride + 整行向量化 + Two-Pass + weight broadcast + FP32 规约。 |
| `exp_groupnorm_001` | 25% | GroupNorm 前向，三 Pass 算法（mean/variance/normalize）+ 大 range autotune + coalesced HW 访问。 |
| `exp_layernorm_fwd_001` | 20% | LayerNorm 前向，persistent kernel + 多算子融合（Norm+Affine+Residual+Swish）+ 三 Pass + heuristic tiling。 |
| `exp_rmsnorm_add_001` | 30% | Add + RMSNorm，指出瓶颈是中间 S 写 HBM 后重载，应保留在寄存器/共享内存复用。 |
| `exp_skip_layernorm_001` | 30% | Skip LayerNorm，residual + affine 融合进 Norm kernel，Two-Pass 处理大 N。 |
| `exp_fused_add_layernorm_001` | 30% | Add + LayerNorm 融合，2D tiling，S 物化到 HBM 以降低寄存器压力，适合大 hidden size。 |
| `exp_fused_norm_rope_001` | 25% | RMSNorm + Partial RoPE 融合，只对 ROPE 维度做旋转，NOPE 维度直接归一化缩放。 |

## 代表 Kernel

选取 `exp_rmsnorm_fused_quant_001`（`189-add_rmsnorm_bias_kernel.py`）作为代表，因为它同时展示了本组最关键的几种内存优化：

- **多算子融合**：Add → RMSNorm → Bias → Quantization 四合一；
- **权重缓存**：`norm_weight_values` 在 grid-stride 循环外加载一次；
- **列方向 tiling + slice 提取**：量化阶段用 `COL_BLOCK_SIZE` 分块，`tl.extract_slice` 避免重载整行；
- **FP32 规约 + 低精度输出**：variance 用 FP32，最终 cast 到 INT8。

```python
@triton.jit
def add_rmsnorm_bias_kernel(
    input_ptr,
    residual_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    quant_scale_ptr,
    quant_offset_ptr,
    output_ptr,
    output2_ptr,
    batch_size,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    valid_mask = cols < hidden_size
    norm_weight_values = tl.load(norm_weight_ptr + cols, mask=valid_mask, other=0.0)
    input_offsets = row_start * hidden_size + cols
    for i in tl.range(row_start, batch_size, row_step):
        # add
        buffered_values = tl.load(input_ptr + input_offsets, mask=valid_mask, other=0.0)
        buffered_values += tl.load(
            residual_ptr + input_offsets, mask=valid_mask, other=0.0
        )
        tl.store(output2_ptr + input_offsets, buffered_values, mask=valid_mask)
        buffered_values = buffered_values.to(tl.float32)
        # norm
        squares = buffered_values * buffered_values  # .to(tl.float32)
        variance = tl.sum(squares) / hidden_size
        reciprocal_std = 1 / tl.sqrt(variance + eps)  # .to(tl.bfloat16)
        buffered_values = buffered_values * reciprocal_std
        buffered_values = buffered_values * norm_weight_values
        # bias
        norm_bias_values = tl.load(norm_bias_ptr + cols, mask=valid_mask, other=0.0)
        buffered_values = buffered_values + norm_bias_values
        if SCALE:
            block_cols = tl.arange(0, COL_BLOCK_SIZE)
            for block_offset in range(0, hidden_size, COL_BLOCK_SIZE):
                col_indices = block_offset + block_cols
                valid_mask2 = col_indices < hidden_size
                block_buffered_values = tl.extract_slice(
                    buffered_values, (block_offset,), (COL_BLOCK_SIZE,), (1,)
                )
                # quant
                quant_scale_values = tl.load(
                    quant_scale_ptr + col_indices, mask=valid_mask2, other=0.0
                )
                quant_offset_values = tl.load(
                    quant_offset_ptr + col_indices, mask=valid_mask2, other=0.0
                )
                block_buffered_values = (
                    block_buffered_values.to(tl.float32) * quant_scale_values
                ) + quant_offset_values
                block_buffered_values = block_buffered_values.cast(
                    tl.int8, overflow_mode="saturate"
                )
                tl.store(
                    output_ptr + i * hidden_size + col_indices,
                    block_buffered_values,
                    mask=valid_mask2,
                )
        else:
            tl.store(output_ptr + input_offsets, buffered_values, mask=valid_mask)

        input_offsets += row_step * hidden_size
```

## 已知实例

- `exp_rmsnorm_backward_001` — Memory Access （收益 25%）
  - 小列 RMSNorm backward（n_cols < 2048），整行向量化、per-pid dW 部分和、LLAMA/GEMMA casting。
  - 对应 kernel: `raw_experiences/extracted_kernels/102-_rms_norm_backward_kernel_no_tiling.py`
- `exp_groupnorm_001` — Memory Access （收益 25%）
  - GroupNorm 前向，三 Pass 算法 + 大 range autotune + coalesced HW 访问。
  - 对应 kernel: `raw_experiences/extracted_kernels/122-group_norm_kernel.py`
- `exp_rmsnorm_npu_001` — Memory Access （收益 35%）
  - 大 hidden size 的 Fused Add RMSNorm，column blocking + Two-Pass + 显式 cache policy。
  - 对应 kernel: `raw_experiences/extracted_kernels/093-_fused_add_rms_norm_forward_kernel_npu.py`
- `exp_layernorm_bwd_001` — Memory Access （收益 25%）
  - 大列 LayerNorm backward（n_cols ≥ 4096），自适应 tiling + FP32 规约 + atomic dW/dB。
  - 对应 kernel: `raw_experiences/extracted_kernels/237-_layernorm_bwd_large_cols_kernel.py`
- `exp_rmsnorm_grad_dx_001` — Memory Access （收益 40%）
  - RMSNorm backward 中 dX，single-pass tiling 把规约和梯度计算合并到一次 UB 内完成。
  - 对应 kernel: `raw_experiences/extracted_kernels/131-rms_norm_grad_dx_kernel.py`
- `exp_rmsnorm_notiling_001` — Memory Access （收益 20%）
  - 小列 RMSNorm 前向（n_cols < 2048），no-tiling + grid-stride + 特定 cache policy。
  - 对应 kernel: `raw_experiences/extracted_kernels/100-_rms_norm_forward_kernel_no_tiling.py`
- `exp_rmsnorm_bwd_large_cols_001` — Memory Access （收益 35%）
  - 大列 RMSNorm backward，Two-Pass + 2D tiling，dW 可选 deterministic 部分和或 atomic add。
  - 对应 kernel: `raw_experiences/extracted_kernels/234-_rmsnorm_bwd_large_cols_kernel.py`
- `exp_rmsnorm_grad_dw_001` — Memory Access （收益 35%）
  - dW 计算，2D tiling + per-program 部分和，避免 atomic 竞争。
  - 对应 kernel: `raw_experiences/extracted_kernels/132-rms_norm_grad_dw_kernel.py`
- `exp_rmsnorm_tiled_001` — Memory Access （收益 25%）
  - 大列 RMSNorm 前向（n_cols ≥ 2048），column tiling + Two-Pass + grid-stride。
  - 对应 kernel: `raw_experiences/extracted_kernels/101-_rms_norm_forward_kernel_tiled.py`
- `exp_gn_backward_001` — Memory Access （收益 35%）
  - GroupNorm backward，空间维度 tiling + 两阶段局部累加，避免原子操作。
  - 对应 kernel: `raw_experiences/extracted_kernels/group_norm_backward_kernel`
- `exp_fused_rms_norm_001` — Memory Access （收益 35%）
  - Add + RMSNorm 融合，residual 原地写回，FP32 variance，Two-Pass 摊平大 N 开销。
  - 对应 kernel: `raw_experiences/extracted_kernels/115-fused_add_rms_norm_kernel.py`
- `exp_rmsnorm_fused_001` — Memory Access （收益 25%）
  - 小列 Fused Add RMSNorm（n_cols < 2048），2D vector loading + 权重预加载。
  - 对应 kernel: `raw_experiences/extracted_kernels/092-_fused_add_rms_norm_forward_kernel_no_tiling.py`
- `exp_layernorm_fwd_001` — Memory Access （收益 20%）
  - LayerNorm 前向，persistent kernel + 多算子融合 + 三 Pass + heuristic tiling。
  - 对应 kernel: `raw_experiences/extracted_kernels/_layernorm_fwd_kernel`
- `exp_rmsnorm_add_001` — Memory Access （收益 30%）
  - Add + RMSNorm，指出中间 S 的冗余重载是瓶颈，应保留在寄存器/共享内存复用。
  - 对应 kernel: `raw_experiences/extracted_kernels/248-_fused_add_rmsnorm_fwd_kernel.py`
- `exp_skip_layernorm_001` — Memory Access （收益 30%）
  - Skip LayerNorm，residual + affine 融合进 Norm kernel，Two-Pass 处理大 N。
  - 对应 kernel: `raw_experiences/extracted_kernels/118-skip_layer_norm_kernel.py`
- `exp_rmsnorm_backward_tiled_001` — Memory Access （收益 25%）
  - 大列 RMSNorm backward，column tiling + Two-Pass + per-pid dW 部分和 + 软件流水。
  - 对应 kernel: `raw_experiences/extracted_kernels/103-_rms_norm_backward_kernel_tiled.py`
- `exp_fused_add_layernorm_001` — Memory Access （收益 30%）
  - Add + LayerNorm 融合，2D tiling，S 物化到 HBM 以降低寄存器压力，适合大 hidden size。
  - 对应 kernel: `raw_experiences/extracted_kernels/250-_fused_add_layernorm_fwd_kernel.py`
- `exp_layernorm_bwd_fused_001` — Memory Access （收益 35%）
  - Fused Add + LayerNorm backward，动态 BLOCK_SIZE_M、循环内 atomic dW/dB、融合 residual gradient。
  - 对应 kernel: `raw_experiences/extracted_kernels/251-_fused_add_layernorm_bwd_kernel.py`
- `exp_fused_split_qk_norm_001` — Memory Access （收益 35%）
  - Split QK + 双 RMSNorm + Copy 融合，输入单遍加载，顺序访问 Q/K-nope/K-pe 三段。
  - 对应 kernel: `raw_experiences/extracted_kernels/191-fused_split_qk_norm_kernel.py`
- `exp_fused_norm_rope_001` — Memory Access （收益 25%）
  - RMSNorm + Partial RoPE 融合，只对 ROPE 维度做旋转，NOPE 维度直接归一化缩放。
  - 对应 kernel: `raw_experiences/extracted_kernels/065-fused_single_norm_and_partial_rope_kernel.py`
- `exp_rmsnorm_fused_quant_001` — Memory Access （收益 50%）
  - Add + RMSNorm + Bias + INT8 Quantization 四合一融合，代表 kernel。
  - 对应 kernel: `raw_experiences/extracted_kernels/189-add_rmsnorm_bias_kernel.py`
- `exp_rmsnorm_bwd_fused_001` — Memory Access （收益 25%）
  - Fused Add + RMSNorm backward，register 级 dW 累加，LLAMA/GEMMA casting mode。
  - 对应 kernel: `raw_experiences/extracted_kernels/249-_fused_add_rmsnorm_bwd_kernel.py`
- `exp_rmsnorm_infer_001` — Memory Access （收益 25%）
  - RMSNorm 推理，grid-stride + Two-Pass + FP32 累加 + tiled coalesced memory access。
  - 对应 kernel: `raw_experiences/extracted_kernels/_rmsnorm_infer_kernel`
- `exp_rmsnorm_001` — Memory Access （收益 15%）
  - 通用 RMSNorm，grid-stride + 整行向量化 + Two-Pass + weight broadcast + FP32 规约。
  - 对应 kernel: `raw_experiences/extracted_kernels/149-_rms_norm_kernel.py`

## 关联知识

- 开发指南: [Normalization 算子优化](../../reference/op-class-normalization.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md) — 主题相关：共享 910b2, 910b3
- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 对应指南: [归一化算子优化](../../reference/op-class-normalization.md) — 主题相关：共享 910b2, 910b3, layernorm, normalization, rmsnorm
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 rmsnorm
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rmsnorm
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 rmsnorm
- 相关主题: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../../ops/rain-fusion-attention.md) — 同属优化知识，主题相关
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 normalization, rmsnorm
- 相关主题: [Rmsnorm Fused Skeleton](../../reference/skeleton-rmsnorm_fused.md) — 同属优化知识，主题相关
- 相关主题: [dl.async_task 实现 Cube/Vector 异构并行](../compute-pipeline/async-task-cube-vector.md) — 同属优化知识，主题相关
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rmsnorm
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, tiling
- 相关主题: [复杂算子拆分：降低编译与调度开销](../compute-pipeline/traj-general_perf-op_split.md) — 主题相关：共享 910b2, 910b3, kernel_fusion
- 相关主题: [Loss 计算优化聚合](../compute-pipeline/traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](../compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, kernel_fusion
- 相关主题: [Normalization Scheduling 优化聚合](../compute-pipeline/traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, normalization, rmsnorm, two_pass
- 相关主题: [Reduction Compute 优化聚合](../compute-pipeline/traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, rmsnorm
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](../compute-pipeline/traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, two_pass
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Activation Backward 内存访问优化聚合](../memory-sram/traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](../memory-sram/traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](../memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](../memory-sram/traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](../memory-sram/traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, rmsnorm, tiling
- 相关主题: [Misc 算子内存访问优化聚合](../memory-sram/traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](../memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](../memory-sram/traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access, two_pass
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](../memory-sram/traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](../memory-sram/traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 相关主题: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](exp-stable-softmax-rmsnorm.md) — 主题相关：共享 rmsnorm
- 相关主题: [FlashAttention 变体的内存访问优化](traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization Compute 优化聚合](traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, layernorm, normalization, rmsnorm
- 相关主题: [跨 tile 归约必须显式两阶段 reduce-then-apply](two-pass-reduce-then-apply.md) — 同属优化知识，主题相关

<!-- okf:related:end -->
