---
title: RoPE（旋转位置编码）算子优化
category: ops
tags: [rope, rotary-embedding, positional-encoding, mrope, partial-rope, backward]
hardware: [910b2, 910b3, 91093, 91095]
last_updated: 2026-04-27
---

# RoPE（旋转位置编码）算子优化

## 概述

RoPE（Rotary Positional Embedding）通过将 token 位置编码嵌入到 Q/K 头的旋转变换中实现位置感知：

```
q_rot[i] = q[i] * cos(θ_pos) - q[i+d/2] * sin(θ_pos)
q_rot[i+d/2] = q[i] * sin(θ_pos) + q[i+d/2] * cos(θ_pos)
```

在 Ascend 910b 上，RoPE 的主要优化目标是：
1. **cos/sin 复用**：同一位置的 cos/sin 在所有 head 间共享，仅加载一次
2. **向量化旋转**：将 head_dim 分成前/后两半（或奇/偶），批量向量化处理
3. **内存对齐**：ALIGNED 路径一次加载整个 rope_dim；非对齐路径分两半加载
4. **Grid-stride loop**：处理可变序列长度，最大化核心利用率

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 经验文件 |
|------|--------|-------|----------|----------|
| 标准 RoPE 前向（通用） | exp_rope_001 | 15% | cos/sin 复用 + persistent grid-stride + constexpr inverse | [yaml](../../raw/experiences/rope/exp_rope_001.yaml) |
| RoPE 前向（NPU 专优） | exp_rope_npu_001 | 35% | cos/sin 复用 + software pipeline (`num_stages`) + 前反向合并 | [yaml](../../raw/experiences/rope/exp_rope_npu_001.yaml) |
| RoPE 前向（内存对齐） | exp_rope_forward_001 | 25% | ALIGNED 路径单次向量加载 + `do_not_specialize=["seq_len"]` | [yaml](../../raw/experiences/rope/exp_rope_forward_001.yaml) |
| RoPE 反向传播 | exp_rope_backward_001 | 25% | grid-stride + IS_VARLEN 变长优化 + HAS_KV_LENS | [yaml](../../raw/experiences/rope/exp_rope_backward_001.yaml) |
| 部分 RoPE（Llama-3 style） | exp_partial_rope_001 | 25% | 跳过 NOPE_DIM + 向量化前/后半加载 + core scheduling | [yaml](../../raw/experiences/rope/exp_partial_rope_001.yaml) |
| Qwen2-VL Multi-RoPE | exp_qwen2vl_mrope_001 | 25% | 时间/高度/宽度三维融合 + head blocking 复用 + software pipeline | [yaml](../../raw/experiences/rope/exp_qwen2vl_mrope_001.yaml) |
| Llama-4 NPU RoPE | exp_rope_llama4_npu_001 | 25% | UB tiling (BLOCK_Q/BLOCK_K) + FMA 指令 + interleaved 复数布局 | [yaml](../../raw/experiences/rope/exp_rope_llama4_npu_001.yaml) |
| SISO 布局 RoPE | exp_rope_siso_001 | 25% | 3D tiling (BLOCK_N×BLOCK_H×BLOCK_D) + 非交错向量化路径 | [yaml](../../raw/experiences/rope/exp_rope_siso_001.yaml) |
| Triton 标准 RoPE | exp_rope_triton_001 | 25% | 向量化 feat_offset 加载 + fused cos/sin 旋转 | [yaml](../../raw/experiences/rope/exp_rope_triton_001.yaml) |
| Neox/非Neox 双路 RoPE | exp_triton_rope_001 | 25% | IS_NEOX_STYLE constexpr 分支消除 + 部分 rope_dim 支持 | [yaml](../../raw/experiences/rope/exp_triton_rope_001.yaml) |

## 关键技术

### 1. cos/sin 一次加载多头复用

cos/sin 表在同一序列位置对所有 head 相同，在 head 循环外加载一次：

```python
@triton.jit
def rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, output_q_ptr, output_k_ptr,
                num_tokens, num_heads_q, num_heads_k, head_dim,
                BLOCK_T: tl.constexpr, HEAD_DIM: tl.constexpr):
    pid = tl.program_id(0)
    token_id = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = token_id < num_tokens

    half = HEAD_DIM // 2
    dim_offs = tl.arange(0, HEAD_DIM)

    # cos/sin：同一位置所有 head 相同，在 head 循环外加载
    cos = tl.load(cos_ptr + token_id[:, None] * HEAD_DIM + dim_offs[None, :],
                  mask=mask_t[:, None])
    sin = tl.load(sin_ptr + token_id[:, None] * HEAD_DIM + dim_offs[None, :],
                  mask=mask_t[:, None])

    for h in range(num_heads_q):
        x1 = tl.load(q_ptr + token_id[:, None] * num_heads_q * HEAD_DIM + h * HEAD_DIM + dim_offs[:half][None, :],
                     mask=mask_t[:, None])
        x2 = tl.load(q_ptr + token_id[:, None] * num_heads_q * HEAD_DIM + h * HEAD_DIM + dim_offs[half:][None, :],
                     mask=mask_t[:, None])
        # 旋转
        o1 = x1 * cos[:, :half] - x2 * sin[:, :half]
        o2 = x1 * sin[:, :half] + x2 * cos[:, :half]
        tl.store(output_q_ptr + ..., o1, mask=mask_t[:, None])
        tl.store(output_q_ptr + ..., o2, mask=mask_t[:, None])
```

### 2. constexpr inverse 消除前/反向分支

`inverse` 标志用 `tl.constexpr` 编译时特化，消除运行时分支：

```python
@triton.jit
def rope_fwd_bwd_kernel(..., INVERSE: tl.constexpr):
    # ...
    if INVERSE:  # 编译时展开，无运行时开销
        o1 = x1 * cos - x2 * (-sin)  # 反向：等价于反向旋转
        o2 = x2 * cos + x1 * (-sin)
    else:
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
```

### 3. 内存对齐 ALIGNED 路径

当 rope_dim 对齐时一次加载，否则分两半：

```python
@triton.jit
def rope_forward_kernel(..., ALIGNED: tl.constexpr, rope_dim, head_dim):
    if ALIGNED:
        # 单次加载完整 rope_dim，最大向量带宽
        x = tl.load(ptr + offs)
    else:
        # 分两半加载（处理非对齐或 rope_dim < head_dim）
        x_low = tl.load(ptr + low_offs, mask=low_mask)
        x_high = tl.load(ptr + high_offs, mask=high_mask)
```

### 4. 部分 RoPE（Partial RoPE，Llama-3 style）

当 `rope_dim < head_dim` 时，只对前 `rope_dim` 维旋转，其余 passthrough：

```python
# 跳过 NOPE_DIM（非旋转部分），从偏移 NOPE_DIM 开始处理
nope_dim = head_dim - rope_dim
half_dim = rope_dim // 2

x_l_offs = base + nope_dim + tl.arange(0, half_dim)      # lower half
x_h_offs = base + nope_dim + half_dim + tl.arange(0, half_dim)  # upper half

x_l = tl.load(ptr + x_l_offs, mask=...)
x_h = tl.load(ptr + x_h_offs, mask=...)

o_l = x_l * cos - x_h * sin
o_h = x_l * sin + x_h * cos
```

### 5. Multi-RoPE（Qwen2-VL）三维位置编码

时间/高度/宽度三维使用不同 cos/sin 表，通过 `tl.where` 动态选择：

```python
# 三维 RoPE：t/h/w 维度各有独立位置编码
@triton.jit
def mrope_kernel(..., HEAD_DIM: tl.constexpr, BLOCK_Q: tl.constexpr):
    dim_idx = tl.arange(0, HEAD_DIM)
    # 按维度判断使用哪套 cos/sin（t/h/w 各占 head_dim/3）
    is_t = dim_idx < HEAD_DIM // 3
    is_h = (dim_idx >= HEAD_DIM // 3) & (dim_idx < 2 * HEAD_DIM // 3)
    # is_w = dim_idx >= 2 * HEAD_DIM // 3

    cos_val = tl.where(is_t, cos_t, tl.where(is_h, cos_h, cos_w))
    sin_val = tl.where(is_t, sin_t, tl.where(is_h, sin_h, sin_w))
```

## 布局变体

| 布局 | 说明 | 加载策略 |
|------|------|----------|
| Neox-style（分块） | `[x_0, ..., x_{d/2-1}, x_{d/2}, ..., x_{d-1}]`，前/后分开 | `IS_NEOX_STYLE=True`：连续分两段加载 |
| GPT-J-style（交错） | `[x_0, x_1, x_2, x_3, ...]`，奇偶交替 | `IS_NEOX_STYLE=False`：步长=2 加载 |
| Partial RoPE | `[passthrough... | rope_rot...]`，仅后 rope_dim 旋转 | 偏移 NOPE_DIM 后处理 |
| SISO | tokens × heads × head_dim 独立分块 | 3D tiling `(BLOCK_N, BLOCK_H, BLOCK_D)` |

## 选择指南

| 场景 | 推荐经验 |
|------|---------|
| 通用 RoPE，910b2/b3 | `exp_rope_npu_001`（最高 35% 提升） |
| 变长序列（varlen batch） | `exp_rope_backward_001`（IS_VARLEN 支持）|
| Llama-3 Partial RoPE | `exp_partial_rope_001` |
| Qwen2-VL Multi-RoPE | `exp_qwen2vl_mrope_001` |
| 需同时支持前/反向 | `exp_rope_npu_001`（`BACKWARD_PASS` constexpr 参数） |
| Llama-4 NPU | `exp_rope_llama4_npu_001`（interleaved complex layout） |

## 常见陷阱

| 陷阱 | 解决方案 |
|------|----------|
| cos/sin 在 head 循环内重复加载 | 提到循环外加载，所有 head 共享（exp_rope_001） |
| rope_dim 与 head_dim 混淆导致越界 | `ALIGNED` constexpr 路径分别处理，加载时用 `mask=offs < rope_dim` |
| int64 token 索引 | 转为 int32 × stride（fp32 偏移），避免 scalar fallback |
| 交错布局误用分块加载 | 用 `IS_NEOX_STYLE` constexpr 编译时选择路径，禁止运行时 if |
| KV cache 偏移未考虑 | `HAS_KV_LENS` 专用路径，用 kv_len 计算 cos/sin 查表偏移 |

## 相关页面

- [ops/elementwise.md](elementwise.md) — RoPE 本质是 elementwise，激活函数反向传播模式可参考
- [ops/attention.md](attention.md) — RoPE 通常在 QKV 投影后、Attention 前执行；Fusion-2 模式
- [ops/fused_ops.md](fused_ops.md) — GEMM + RMSNorm + RoPE 三合一融合（Fusion-2）
- [techniques/ilp.md](../techniques/ilp.md) — cos/sin 加载的 for-loop tiling + software pipeline
- [techniques/data_types.md](../techniques/data_types.md) — int64 索引退化问题

## 被以下页面引用

- [ops/attention.md](attention.md) — Attention 前的 QK RoPE 编码步骤
- [ops/fused_ops.md](fused_ops.md) — Fusion-2 Linear+RMSNorm+RoPE
- [experience_index.md](../experience_index.md) — RoPE 区全量经验 ID
