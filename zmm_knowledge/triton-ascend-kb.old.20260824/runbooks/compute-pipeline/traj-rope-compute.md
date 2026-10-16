---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: rope
dimension: "2.1"
title: RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合
description: 聚合 6 条 RoPE 在 Ascend 910B 上的 compute 经验，覆盖 cos/sin 跨 head 复用、Neox/interleaved/partial 向量化旋转、grid-stride 持久调度，以及 Qwen2-VL M-RoPE 多模态位置编码融合，收益 15%–25%。
tags: [rope, compute, cos_sin_reuse, vectorized_rotation, partial_rope, mrope, grid_stride, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/rope/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合

## 概述

RoPE（Rotary Position Embedding）在 LLM/VLM 中每次 forward/decode 都要对 Q/K 做一次二维旋转，**计算密度低、内存访问模式复杂**。Ascend 910B 上它的主要瓶颈不是矩阵乘，而是：
- 每个 head 重复读取同一份 cos/sin，造成冗余 HBM 流量；
- 标量或半标量的逐元素旋转无法充分利用 Vector SIMD；
- Neox 分块布局、交错（interleaved）布局、Partial RoPE、M-RoPE 等不同变体对偏移计算和掩码写法要求不同，容易写出带运行时分支或 i64 指针运算的代码。

本卡聚合 6 条 compute experience，覆盖两个层面：
1. **通用 RoPE 加速**：cos/sin 复用、向量化 chunked rotation、partial RoPE、grid-stride 调度；
2. **专用变体融合**：Qwen2-VL 的 M-RoPE 把 t/h/w 三组位置编码直接融进一个 kernel，避免预处理和多次启动。

**核心 trade-off**：
- 复用 cos/sin 能降低内存带宽，但会把小表固定在寄存器，head_dim 很大时可能挤占 tile 空间；
- `tl.constexpr` 特化可消除运行时分支，但组合过多（forward/backward × Neox/interleaved × partial）会增加编译产物体积；
- M-RoPE 融合减少 kernel 启动和中间张量，但 3D 表指针计算更复杂，容易引入 scalar 路径。

## 共同模式

### 1. cos/sin 跨 head 复用

**问题根因**：同一 token 位置的位置编码对所有 attention head 都相同。如果按 `for head: load cos/sin + rotate` 写，每个 head 都把同一段 embedding 从 HBM 读一遍，带宽随 head 数线性增长。

**收益明显条件**：
- head 数较多（如 32/64/128），cos/sin 表相对较小；
- batch × seq 规模不大，embedding 能被寄存器/SRAM 留住。

**实现要点**：
- 在外层循环按 token/block 加载 cos/sin；
- 内层循环遍历 head 时直接使用已加载的向量；
- M-RoPE 场景下分别加载 t/h/w 三组表，再用 `tl.where` 按维度索引选择。

**常见坑**：
- 不要把 cos/sin 加载放到 head 循环内部；
- head_dim 极大时，寄存器压力可能反而升高，需要把 head 再分块（BLOCK_Q/BLOCK_K）。

### 2. 向量化 chunked rotation

**问题根因**：RoPE 的数学形式是复数乘法 `x₁·cos − x₂·sin`，但标量实现会让 Vector 单元大量空闲，且 load/store 无法合并。

**收益明显条件**：
- head_dim 能被 64/128 等向量宽度整除；
- 数据按 Neox（chunked，前一半是 x₁、后一半是 x₂）或 interleaved 排列，且能一次性构造连续偏移。

**实现要点**：
- Neox 布局：把 `head_dim // 2` 作为连续块加载，前块为 x₁、后块为 x₂；
- Interleaved 布局：用 `tl.arange(0, rope_dim, 2)` 和 `+1` 取相邻 pair；
- 一次 load 完成 `x₁`、`x₂`， fused 计算 `x₁*cos − x₂*sin` 和 `x₂*cos + x₁*sin`，再 store。

**常见坑**：
- 布局判断错误会导致旋转对象错位，模型输出直接错乱；
- Partial RoPE 的非旋转段（NOPE_DIM）必须跳过，不能混进旋转计算。

### 3. Partial RoPE：跳过非旋转维度

**问题根因**：Llama-3 等模型只把头维中的一部分（rope_dim）做旋转，其余维度原样透传。对整个 head_dim 做旋转会浪费约 `1 − rope_dim/head_dim` 的无效计算。

**收益明显条件**：
- `rope_dim < head_dim`，差距越大收益越高；
- 编译期能确定 NOPE_DIM 和 ROPE_DIM。

**实现要点**：
- 从 `NOPE_DIM` 开始构造旋转偏移；
- 透传段和旋转段分别用不同的 load/store 路径，或在同一 kernel 中用 mask 分段；
- 保持 cos/sin 只覆盖 `ROPE_DIM // 2` 长度。

**常见坑**：
- 不要把 NOPE_DIM 也纳入 `tl.arange(0, head_dim // 2)`，否则会旋转不应旋转的维度；
- 边界 mask 要同时覆盖 rope_dim 和 head_dim 两个范围。

### 4. Grid-stride / persistent 调度

**问题根因**：RoPE 的输入形状随 batch、seq_len 变化。若 grid size 与 seq_len 或 token 数绑定，不同 batch 会反复触发 JIT 重新编译，且小 batch 容易空载。

**收益明显条件**：
- 服务化推理中 batch 和 seq_len 多变；
- token/block 数量不是核心数的整数倍。

**实现要点**：
- 固定 grid size 为硬件核心数；
- 用 `for seq_block_id in range(pid, NUM_BLOCKS, NUM_CORES)` 让每个 program 处理多个 block；
- `total_rows`、`num_tokens` 等作为运行时参数传入。

**常见坑**：
- block 切得太细会削弱 cos/sin 复用效果；
- 持久循环内不要保留过多跨迭代状态，否则寄存器 spill。

### 5. M-RoPE 多模态位置编码融合

**问题根因**：Qwen2-VL 等 VLM 对 t/h/w 三个模态分别做 RoPE。朴素做法是先拆 tensor、分别计算、再拼接，产生多次 kernel 启动和中间内存读写。

**收益明显条件**：
- 多模态模型，输入已经把 t/h/w 分段放在 head_dim 内；
- 三个段的 cos/sin 表能在 SRAM 同时驻留。

**实现要点**：
- 在 kernel 内按 `mrope_section_t / mrope_section_h / mrope_section_w` 计算三组 cos/sin 指针；
- 用 `tl.where` 根据维度索引选择对应模态的 cos/sin，避免分支；
- Q/K head 用 `BLOCK_Q / BLOCK_K` 分块，复用已选好的 cos/sin 向量。

**常见坑**：
- t/h/w 三段长度必须严格对齐模型配置；
- `tl.where` 的三个分支都要计算，不要写成 Python `if/else`，否则向量化失败。

### 6. constexpr 特化消除运行时分支

**问题根因**：forward/backward、Neox/interleaved、partial/full、inverse 等控制流如果在运行时判断，会导致 warp 内分支发散和标量比较。

**收益明显条件**：
- 所有控制参数在编译期已知；
- 可以接受为不同组合生成多个 kernel 变体。

**实现要点**：
- `IS_NEOX_STYLE`、`inverse`、`BACKWARD_PASS` 等用 `tl.constexpr`；
- cos/sin 计算前 cast 到 fp32，保证数值稳定；
- 对 backward 的符号翻转单独写 constexpr 分支：`new_left = x1*cos ± x2*sin`。

**常见坑**：
- constexpr 组合爆炸会增加编译时间和二进制大小；
- 不要把运行时变量（如 seq_len）误设为 constexpr。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_triton_rope_001` | 25% | 通用 Triton RoPE：同时支持 Neox/interleaved 双布局、partial RoPE 与位置索引/预计算 cos/sin，向量化 chunked 处理。 |
| `exp_rope_001` | 15% | SRAM 内 RoPE 微内核：cos/sin 按 block 复用、grid-stride 调度、复数乘拆解为浮点运算，需配合上层内存加载策略。 |
| `exp_rope_siso_001` | 25% | SISO 非交错布局：3D tiling (BLOCK_N/BLOCK_H/BLOCK_D)，非交错路径一次性向量化整个 head_dim。 |
| `exp_partial_rope_001` | 25% | Partial RoPE：从 NOPE_DIM 开始偏移，只旋转 rope_dim 段，显式核心调度。 |
| `exp_qwen2vl_mrope_001` | 25% | Qwen2-VL M-RoPE：t/h/w 三模态位置编码在一个 kernel 内融合，head 分块复用 cos/sin。 |
| `exp_rope_triton_001` | 25% | LLM 通用 RoPE：多布局自适应、向量化 pair loading、constexpr 特化、混合精度旋转。 |

## 代表 Kernel

选择 `exp_qwen2vl_mrope_001` 作为代表，因为它集中展示了 **M-RoPE 三模态选择、cos/sin 复用、head 分块、向量化旋转** 四个关键模式，代码结构对其它 RoPE 变体也有借鉴意义。

```python
@triton.jit
def _triton_qwen2vl_mrope_npu(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    sin,
    sl,
    bs: tl.constexpr,
    total_rows: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_per_program = (total_rows + num_programs - 1) // num_programs
    start_row = program_id * rows_per_program
    actual_rows = tl.minimum(rows_per_program, total_rows - start_row)

    # grid-stride 风格：每个 program 处理多行
    for row_offset in tl.range(0, actual_rows, num_stages=NUM_STAGES):
        pid = start_row + row_offset

        t_end = mrope_section_t
        h_end = t_end + mrope_section_h

        # 三个模态的 cos/sin 指针
        t_cos = cos + pid * hd
        h_cos = t_cos + bs * sl * hd
        w_cos = h_cos + bs * sl * hd
        t_sin = sin + pid * hd
        h_sin = t_sin + bs * sl * hd
        w_sin = h_sin + bs * sl * hd

        q_base = q_ptr + pid * q_row_stride
        k_base = k_ptr + pid * k_row_stride

        d_idx = tl.arange(0, hd // 2)
        d_mask = d_idx < (hd // 2)

        pos_mask_t = d_idx < t_end
        pos_mask_h = (d_idx >= t_end) & (d_idx < h_end)

        # 按模态加载并选择
        text_cos_vals = tl.load(t_cos + d_idx, mask=d_mask, other=0)
        text_sin_vals = tl.load(t_sin + d_idx, mask=d_mask, other=0)
        height_cos_vals = tl.load(h_cos + d_idx, mask=d_mask, other=0)
        height_sin_vals = tl.load(h_sin + d_idx, mask=d_mask, other=0)
        width_cos_vals = tl.load(w_cos + d_idx, mask=d_mask, other=0)
        width_sin_vals = tl.load(w_sin + d_idx, mask=d_mask, other=0)

        cos_vals = tl.where(pos_mask_t, text_cos_vals,
                            tl.where(pos_mask_h, height_cos_vals, width_cos_vals))
        sin_vals = tl.where(pos_mask_t, text_sin_vals,
                            tl.where(pos_mask_h, height_sin_vals, width_sin_vals))

        # Q head 分块处理，复用同一组 cos/sin
        for qh_block in range(0, n_qh, BLOCK_Q):
            qh_idx = tl.arange(0, BLOCK_Q) + qh_block
            qh_mask = qh_idx < n_qh

            block_mask = qh_mask[:, None] & d_mask[None, :]
            offsets = qh_idx[:, None] * hd + d_idx[None, :]

            q_left = tl.load(q_base + offsets, mask=block_mask, other=0)
            q_right = tl.load(q_base + offsets + (hd // 2), mask=block_mask, other=0)

            if not BACKWARD_PASS:
                new_left = q_left * cos_vals - q_right * sin_vals
                new_right = q_right * cos_vals + q_left * sin_vals
            else:
                new_left = q_left * cos_vals + q_right * sin_vals
                new_right = q_right * cos_vals - q_left * sin_vals

            tl.store(q_base + offsets, new_left, mask=block_mask)
            tl.store(q_base + offsets + (hd // 2), new_right, mask=block_mask)

        # K head 分块处理
        for kh_block in range(0, n_kh, BLOCK_K):
            kh_idx = tl.arange(0, BLOCK_K) + kh_block
            kh_mask = kh_idx < n_kh

            block_mask = kh_mask[:, None] & d_mask[None, :]
            offsets = kh_idx[:, None] * hd + d_idx[None, :]

            k_left = tl.load(k_base + offsets, mask=block_mask, other=0)
            k_right = tl.load(k_base + offsets + (hd // 2), mask=block_mask, other=0)

            if not BACKWARD_PASS:
                new_left = k_left * cos_vals - k_right * sin_vals
                new_right = k_right * cos_vals + k_left * sin_vals
            else:
                new_left = k_left * cos_vals + k_right * sin_vals
                new_right = k_right * cos_vals - k_left * sin_vals

            tl.store(k_base + offsets, new_left, mask=block_mask)
            tl.store(k_base + offsets + (hd // 2), new_right, mask=block_mask)
```

**关键实现细节**：
- `total_rows` 运行时传入，`rows_per_program` 由程序自己计算，实现 grid-stride 负载均衡；
- t/h/w 三组 cos/sin 通过 `tl.where` 二维选择合并为一条向量化路径，无 Python 分支；
- `BACKWARD_PASS` 用 `tl.constexpr` 特化，编译期确定符号翻转；
- Q/K head 用 `BLOCK_Q/BLOCK_K` 分块，复用同一行 cos/sin，显著降低 HBM 流量。

## 已知实例

- `exp_triton_rope_001` — Compute（收益 25%）
  - 通用 Triton RoPE，支持 Neox/interleaved 双布局、partial RoPE、预计算/位置索引 cos/sin。
- `exp_rope_001` — Compute（收益 15%）
  - SRAM 内 RoPE 微内核：cos/sin 复用、grid-stride 调度、复数乘拆解为浮点运算，需配合上层加载策略。
- `exp_rope_siso_001` — Compute（收益 25%）
  - SISO 非交错布局全向量化，BLOCK_D 对齐硬件向量宽度。
- `exp_partial_rope_001` — Compute（收益 25%）
  - Partial RoPE：跳过 NOPE_DIM，只旋转 rope_dim 段，显式 NUM_CORES 调度。
- `exp_qwen2vl_mrope_001` — Compute（收益 25%）
  - Qwen2-VL 3D 位置编码融合，head 分块复用 cos/sin，代表 kernel 见上。
- `exp_rope_triton_001` — Compute（收益 25%）
  - LLM 通用 RoPE：多布局自适应、向量化 pair loading、constexpr 特化、混合精度旋转。

## 关联知识

- 开发指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, rope
- 对应指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md) — 主题相关：共享 910b2, 910b3, mrope, partial_rope, rope
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 rope
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rope
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 rope
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 rope
- 相关主题: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 主题相关：共享 rope
- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, cos_sin_reuse, rope
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](../portability/traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 rope
- 相关主题: [persistent kernel 模式（grid-stride 摊还启动开销）](persistent-kernel-pattern.md) — 主题相关：共享 grid_stride
- 相关主题: [SwiGLU 计算优化：门控融合、展平布局与激活量化](traj-activation-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [Attention Scheduling 优化聚合](traj-attention-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [大词表 Entropy Loss Backward 调度融合优化](traj-loss-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Scheduling 优化聚合](traj-normalization-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, grid_stride, rope
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
