---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: activation
dimension: "2.1"
title: SwiGLU 计算优化：门控融合、展平布局与激活量化
description: 聚合 5 条 activation compute 经验，覆盖 SwiGLU 前向门控融合、OpenAI 交错布局、反向梯度分离、展平 1D 布局与激活后量化融合，收益 25%–35%。
tags: [activation, compute, swiglu, gated_activation, quantization, moe, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/activation/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# SwiGLU 计算优化：门控融合、展平布局与激活量化

## 概述

SwiGLU 及其门控变体（Gated Linear Unit、GDN 门控）在现代 LLM 的 FFN/MoE 中广泛使用。其计算形式为：

```
out = gate * SiLU(gate) * up
```

虽然每个操作都是逐元素计算，但在 Ascend 910B 上：
- 若拆成多个独立 kernel，会多次读写 HBM，成为内存墙瓶颈；
- 不同权重排布（交错 `gate, up, gate, up…` vs 连续两段 `gate|up` vs 展平 1D）需要不同的索引与向量化策略；
- MoE 场景下还需要按 `group_list` 动态分块，避免 pad 浪费。

本卡 5 条 experience 覆盖了 SwiGLU 的前向、反向、量化融合与 MoE 调度优化。

**核心 trade-off**：
- 融合越多，HBM 读写越少，但 kernel 内寄存器压力和编译复杂度越高；
- 交错布局可复用向量化 load，但需要 `tl.extract_slice` 拆分 gate/up；
- 展平布局省掉 reshape，但要求调用方保证连续 1D；
- 激活后量化能省掉一次 fp16/bf16 输出往返，但需在线计算 scale 并做饱和 cast。

## 共同模式

### 1. 门控激活融合：把 gate、SiLU、up 乘放在同一 kernel

**问题根因**：`gate * SiLU(gate) * up` 三个逐元素操作如果拆成 2–3 个 kernel，会产生中间张量写回 HBM 再读取的开销，对 elementwise 算子这是主要瓶颈。

**收益明显条件**：
- FFN/MoE 的 SwiGLU 前向是热点；
- gate 和 up 已经同时可用；
- 输出需要直接参与后续 matmul 或量化。

**实现要点**：
```python
x1 = tl.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
x2 = tl.extract_slice(cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,))
out = x1 * tl.sigmoid(x1) * x2
```
- 对交错布局用 `extract_slice` 拆 gate/up；
- 对连续布局直接用两段指针；
- 反向时分别计算 gate 和 up 的梯度，避免重复加载上游梯度。

**常见坑**：
- `extract_slice` 的 offsets/sizes 必须对应当前 BLOCK 的排布；
- SiLU 的数值稳定性：大负数时 `sigmoid` 会下溢，必要时用 `tl.sigmoid` 内置保护；
- 不要把 matmul 也硬塞进同一 kernel，保留 `tl.dot` 的 Cube 加速。

### 2. 交错布局 vs 展平布局选择

**问题根因**：不同模型对 SwiGLU 权重的排布不同：OpenAI 格式常用交错 `gate, up, gate, up…`，而常规实现常用连续两段 `gate|up` 或完全展平 1D。索引方式错误会导致向量化失效或内存不对齐。

**收益明显条件**：
- 输入已经是交错或展平布局，且不能简单 `contiguous()` 转换；
- 大张量、向量化带宽是瓶颈；
- 需要和上游/下游 kernel 保持同一 layout 以避免 transpose。

**实现要点**：
- **交错布局**：把 `TOTAL_COLS` 视为 `(HALF_COLS, 2)`，用 `extract_slice` 取 gate/up；
- **连续两段布局**：指针直接偏移 `HALF_COLS`；
- **展平 1D**：按 `row * TOTAL_COLS + col` 一维遍历，适合 `hidden_states` 已经被 flatten 的场景。

**常见坑**：
- 混淆 OAI 格式与普通 SwiGLU 的权重分割点；
- 展平布局下 stride 计算错误会导致 gate/up 错位；
- 不要为了统一 layout 而做不必要的 `reshape`/`permute`。

### 3. 反向梯度分离

**问题根因**：SwiGLU 反向需要同时计算 gate 梯度 `d_gate` 和 up 梯度 `d_up`，两者共享上游梯度 `d_out`。若分两次 kernel 分别计算，会重复加载 `d_out`、`gate`、`up`。

**收益明显条件**：
- SwiGLU backward 是训练热点；
- gate 和 up 张量较大，重复 load 成本高；
- 输出 layout 允许同时写回两段梯度。

**实现要点**：
- 一次加载 `gate`、`up`、`d_out`；
- 用 SiLU 导数公式分别计算 `d_gate` 和 `d_up`；
- 对交错输出，用 `tl.interleave` 或直接分段 store。

**常见坑**：
- `d_gate` 中 SiLU 导数计算容易遗漏 `gate * sigmoid(gate)` 项；
- 两段梯度写回时的 mask 要分别处理；
- 数值精度：反向建议用 fp32 累加再 cast。

### 4. 激活后立即量化

**问题根因**：MoE 等场景下 SwiGLU 输出需要以 int8 送入下一层。如果先写回 fp16 再启动量化 kernel，会多一次 HBM 往返。

**收益明显条件**：
- 后续算子支持 int8 输入；
- SwiGLU 输出元素数大；
- 量化 scale 可按 row/expert 计算。

**实现要点**：
```python
scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
    tmp_out = tl.extract_slice(out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,))
    tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
    tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")
    tl.store(out_ptr + o_offsets, tmp_out, mask=mask)
```
- 在寄存器中计算 `scale`；
- 按 `COL_BLOCK_SIZE` 分块做饱和 int8 cast；
- 支持可选 `SCALE` 路径：不需要量化时直接存 fp16。

**常见坑**：
- `DTYPE_MAX` 要根据目标 int8 范围（127）设置；
- 量化 scale 的 reduce 要覆盖当前 row/expert 的所有元素；
- 饱和 cast 与 round 模式需与训练框架一致。

### 5. MoE group_list  minibatch 分块

**问题根因**：MoE 中每个 expert 处理的 token 数不同，直接按 `num_tokens * top_k` 启动会导致大量空跑 program。用 `group_list`（cumsum）把有效 token 连续化，再按核心数均匀分块。

**收益明显条件**：
- MoE layer，每个 expert 的 token 数动态变化；
- `group_list` 已经提供 cumsum；
- 需要避免 padding token 占用计算资源。

**实现要点**：
```python
if GROUP_LIST_TYPE == 0:  # cusum
    total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
else:
    group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
    total_rows = tl.sum(group_list)

block_size = (total_rows - 1) // NUM_CORES + 1
pid = tl.program_id(0)
row_begin = pid * block_size
if row_begin >= total_rows:
    return
row_end = tl.minimum((pid + 1) * block_size, total_rows)
```
- 先通过 `group_list` 拿到实际有效行数；
- 每个 program 负责一段连续行，避免稀疏索引；
- 无效 program 早退。

**常见坑**：
- `group_list` 类型（cusum vs 直方图）要统一；
- `block_size` 为 0 时要做保护；
-  expert 并行和 token 并行的边界要分清。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_swiglu_oai_001` | 25% | OpenAI 格式 SwiGLU 前向：输入 gate/up 交错排布，用 `extract_slice` 拆分后做门控乘，支持 `gemm1_alpha` 缩放。 |
| `exp_fused_gating_001` | 25% | 通用门控激活（GLU/GDN 变体）融合计算：把门控、激活、乘法门控合并为单个 kernel，减少 elementwise 往返。 |
| `exp_swiglu_bwd_001` | 25% | SwiGLU 反向：一次加载 gate/up/d_out，分别计算 gate 与 up 梯度，避免重复 load。 |
| `exp_swiglu_bwd_flat_001` | 35% | 展平 1D 布局的 SwiGLU 反向：按一维连续索引遍历，适合已经 flatten 的 activation 层，缓解内存 bound。 |
| `exp_swiglu_quant_001` | 35% | SwiGLU 前向 + 立即量化融合：门控计算后直接在寄存器内求 scale 并饱和 cast 到 int8，MoE 场景收益显著。 |

## 代表 Kernel

选择 `exp_swiglu_quant_001` 作为代表，因为它同时覆盖**门控激活融合、激活后量化、MoE group_list 分块**三个核心模式，且收益最高（35%）。

```python
@triton.jit
def _swiglu_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
):
    # 通过 group_list 计算实际有效行数
    if GROUP_LIST_TYPE == 0:  # cusum
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
    else:
        gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
        gl_mask = gl_offsets < NUM_EXPERTS
        group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
        total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # 交错布局：一次性加载 gate+up
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        x1 = tl.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        x2 = tl.extract_slice(cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,))

        # 门控激活融合：gate * SiLU(gate) * up
        out = x1 * tl.sigmoid(x1) * x2

        # 可选：激活后立即量化
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp_out = tl.extract_slice(
                    out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,)
                )
                tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
                tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")

                o_offsets = row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(x_ptr.dtype.element_ty))
```

**关键实现细节**：
- `group_list` 决定实际工作负载，无效 program 早退；
- 交错布局下用 `extract_slice` 拆分 gate/up；
- `SiLU(gate) * gate * up` 在一个 kernel 内完成，无中间 HBM 往返；
- 量化路径在寄存器内计算 scale 并饱和 cast 到 int8；
- `SCALE` 用 `tl.constexpr` 特化量化与非量化路径。

## 已知实例

- `exp_swiglu_oai_001` — Compute（收益 25%）
  - OpenAI 格式 SwiGLU 前向：gate/up 交错排布，支持 `gemm1_alpha` 缩放。对应 kernel: `raw_experiences/extracted_kernels/178-swiglu_oai_kernel.py`。
- `exp_fused_gating_001` — Compute（收益 25%）
  - 通用门控激活融合：GLU/GDN 变体的门控、激活与乘法合并为单 kernel。对应 kernel: `raw_experiences/extracted_kernels/163-fused_gdn_gating_kernel.py`。
- `exp_swiglu_bwd_001` — Compute（收益 25%）
  - SwiGLU 反向：分别计算 gate 与 up 梯度，避免重复加载上游梯度。对应 kernel: `raw_experiences/extracted_kernels/212-_swiglu_bwd_kernel.py`。
- `exp_swiglu_bwd_flat_001` — Compute（收益 35%）
  - 展平 1D 布局的 SwiGLU 反向，适合已经 flatten 的内存排布。对应 kernel: `raw_experiences/extracted_kernels/106-_swiglu_backward_kernel_flat.py`。
- `exp_swiglu_quant_001` — Compute（收益 35%）
  - SwiGLU 前向 + 立即量化融合，支持 MoE group_list 动态分块。对应 kernel: `raw_experiences/extracted_kernels/146-_swiglu_quant_kernel.py`。

## 关联知识

- 开发指南: [Elementwise 算子优化](../../reference/op-class-elementwise.md)

<!-- okf:related:start -->

# 相关

- 相关主题: [Attention 计算优化：Online Softmax、Split-K、Sparse Attention 与混合精度](../algorithm-numerics/traj-attention-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Normalization Compute 优化聚合](../algorithm-numerics/traj-normalization-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Activation Backward 内存访问优化聚合](../memory-sram/traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, activation, swiglu
- 相关主题: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../meta-skills/quant-fusion-int8-precision-diagnosis.md) — 主题相关：共享 quantization
- 相关主题: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3
- 相关主题: [算子融合模式与约束决策](operator-fusion-patterns.md) — 主题相关：共享 activation, quantization
- 相关主题: [Activation Scheduling 优化聚合](traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, activation
- 相关主题: [Causal Conv1D Compute 优化聚合](traj-conv-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Scatter Weight Gradient 向量化计算优化](traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Loss 计算优化聚合](traj-loss-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Matmul Compute 优化聚合](traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, compute, quantization
- 相关主题: [Misc 算子计算优化：向量化、微内核与阶段融合](traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [MoE 稀疏 Weight Gradient 计算优化](traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, compute, moe
- 相关主题: [Reduction Compute 优化聚合](traj-reduction-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [Softmax 计算优化：Online Softmax、Two-Pass 向量化与数值稳定](traj-softmax-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [三角矩阵求解（solve_tril）计算优化：结构零元掩码消除与 constexpr 双路径](traj-solve_tril-compute.md) — 主题相关：共享 910b2, 910b3, compute
- 相关主题: [SSM Compute 优化聚合](traj-ssm-compute.md) — 主题相关：共享 910b2, 910b3, compute

<!-- okf:related:end -->
