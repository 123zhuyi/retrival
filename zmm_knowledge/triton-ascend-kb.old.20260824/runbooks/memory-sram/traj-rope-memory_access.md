---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: rope
dimension: "1.1"
title: RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化
description: 聚合 2 条 rope memory access 经验，覆盖 NPU 专用 RoPE 的 cos/sin 复用、软件流水线与 head 分块，以及通用 RoPE 前向的 ALIGNED 单次加载、非对齐分半加载与 `do_not_specialize` 动态特化，收益 25%–35%。
tags: [rope, memory_access, cos_sin_reuse, aligned_load, head_blocking, software_pipeline, do_not_specialize, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/rope/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化

## 概述

RoPE 的内存访问瓶颈不在旋转计算本身，而在如何高效加载 Q/K 和 cos/sin 表：
1. **cos/sin 复用**：同一序列位置的所有 head 共享同一组 cos/sin，若在 head 循环内重复加载会浪费大量带宽；
2. **对齐加载**：`rope_dim` 对齐时一次性加载完整旋转维度，非对齐时需分前后两半；
3. **动态特化**：推理时 `seq_len` 变化，用 `do_not_specialize` 避免重复编译；
4. **head 分块 + 软件流水线**：head 数多时在内存访问层做 head blocking，并用 `num_stages` 掩盖延迟。

本卡 2 条 experience 分别覆盖：
- `exp_rope_npu_001`（35%）：NPU 专用路径，强调 cos/sin 复用、软件流水线、head 分块；
- `exp_rope_forward_001`（25%）：通用前向路径，强调 ALIGNED 单次加载、`do_not_specialize=["seq_len"]`。

**核心 trade-off**：
- cos/sin 复用要求它们在 head 循环外驻留寄存器/UB，head 分块过小会导致 cos/sin 被多次加载；
- ALIGNED 路径带宽最高，但要求 `rope_dim` 满足对齐条件；
- `do_not_specialize` 减少编译缓存，但会损失部分静态 shape 优化。

## 共同模式

### 1. cos/sin 复用：head 循环外加载一次

**问题根因**：cos/sin 表只与位置有关，与 head 无关。如果每个 head 单独加载，会重复读取同一段内存，尤其是在 head 数多（如 32/64/128）时带宽浪费严重。

**收益明显条件**：
- `n_qh`、`n_kh` 较大；
- 序列长度和 head_dim 使得 cos/sin tile 可以驻留 UB；
- 前向/反向都需要加载 cos/sin。

**实现要点**：
```python
# 在 head 循环外加载当前 token block 的 cos/sin
cos_block = tl.load(
    cos_token_ptr + half_dim_offsets[None, :],
    mask=seq_mask[:, None] & half_dim_mask[None, :],
    other=0.0,
)
sin_block = tl.load(
    sin_token_ptr + half_dim_offsets[None, :],
    mask=seq_mask[:, None] & half_dim_mask[None, :],
    other=0.0,
)

# 复用 cos/sin tile 处理所有 Q/K heads
for h_block in range(0, n_qh, BLOCK_H):
    ...
```
- cos/sin 按 `(TOKEN_BLOCK_SIZE, half_rope_dim)` 形状加载；
- reshape 成 `(TOKEN_BLOCK_SIZE, 1, half_rope_dim)` 广播到所有 head；
- head 循环内只加载 Q/K，不重复加载 cos/sin。

**常见坑**：
- cos/sin 的 dtype 要与 Q/K 匹配，否则 broadcast/转类型引入额外开销；
- cos/sin tile 过大导致 UB spill，需结合 head 分块；
- 反向时 cos 不变、sin 符号取反，不要重复加载另一张表。

### 2. ALIGNED 路径：单次加载完整 rope_dim

**问题根因**：当 `rope_dim` 满足对齐条件时，可以从 `nope_dim` 开始一次性加载 `rope_dim` 个元素，向量化程度和带宽最高。

**收益明显条件**：
- `rope_dim` 是向量宽度的整数倍；
- `nope_dim` 到 `nope_dim + rope_dim` 的地址对齐；
- 绝大多数推理 shape 满足对齐。

**实现要点**：
```python
rope_dim_offsets = tl.arange(0, rope_dim)
q_offsets = (
    batch_idx * q_batch_stride
    + global_seq_offsets[:, None, None] * q_seq_stride
    + head_q_offsets[None, :, None] * hd
    + nope_dim
    + rope_dim_offsets[None, None, :]
)
q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
q_tile = _compute_rope(q_tile, sin_tile, cos_tile, n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, False)
tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)
```
- `tl.arange(0, rope_dim)` 连续索引；
- 一次 load 得到完整旋转区；
- mask 用 `rope_dim_mask` 兜底。

**常见坑**：
- 把 `rope_dim` 当 `head_dim` 使用导致越界；
- 非对齐场景强行走 ALIGNED 路径会读取或写入非法位置；
- ALIGNED 判断应使用 `tl.constexpr` 编译期分支。

### 3. 非 ALIGNED 路径：分前后两半加载

**问题根因**：当 `rope_dim` 不是向量宽度整数倍，或模型使用 Partial RoPE（`rope_dim < head_dim`）时，需要把旋转区分成前后两半分别加载，避免越界和不规则访问。

**收益明显条件**：
- `rope_dim` 不对齐或小于 `head_dim`；
- 使用 Partial RoPE（如 Llama-3 style）；
- 需要精确控制 mask。

**实现要点**：
```python
q_offsets_half1 = base + nope_dim + half_dim_offsets[None, None, :]
q_offsets_half2 = base + nope_dim + half_rope_dim + half_dim_offsets[None, None, :]
q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0)
q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0)
new_q_1, new_q_2 = _compute_rope_separated(q_tile_1, q_tile_2, sin_tile, cos_tile, False)
tl.store(q_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
tl.store(q_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)
```
- 分别加载 `[nope_dim, nope_dim+half)` 和 `[nope_dim+half, nope_dim+rope_dim)`；
- 两段各自走半维度旋转；
- store 时也分两次，mask 独立。

**常见坑**：
- 前后两半的 `half_dim_offsets` 范围重复或遗漏；
- 非对齐尾部未 mask 导致写入 `nope_dim + rope_dim` 之后的 passthrough 区；
- `_compute_rope_separated` 的输入 shape 要与 `cos_tile` 广播匹配。

### 4. Head 分块：控制 UB 工作集

**问题根因**：当 head 数多、token block 大时，一次性加载所有 head 的 Q/K 会超出 UB。按 `BLOCK_H` 对 head 分块，让活跃工作集保持在 UB 内。

**收益明显条件**：
- `n_qh`/`n_kh` ≥ 32；
- `TOKEN_BLOCK_SIZE * BLOCK_H * rope_dim` 接近或超过 UB 容量；
- 需要与 cos/sin 复用结合，避免分块后重复加载 cos/sin。

**实现要点**：
- 外层循环 head block，内层加载 `BLOCK_H` 个 head；
- cos/sin tile 在 head 循环外加载并 reshape 广播；
- `BLOCK_H` 选择应使 `TOKEN_BLOCK_SIZE * BLOCK_H * rope_dim * dtype_size` 小于 UB。

**常见坑**：
- `BLOCK_H` 不是 `n_qh` 的约数时，最后一个 block 需 mask；
- head 维 stride 计算错误（注意 `hd` 与 `q_head_stride` 的区别）；
- 分块后 cos/sin 广播维度要与 head block 对齐。

### 5. 软件流水线：`num_stages` 掩盖 load-store 延迟

**问题根因**：RoPE 是内存 bound 操作，head 循环内连续 load→compute→store 会产生 MTE2/Vector 流水线空泡。

**收益明显条件**：
- head 循环次数多；
- 每次循环的 tile shape 固定；
- 后端支持 software pipeline（Ascend NPU 支持）。

**实现要点**：
```python
@triton.jit
def _triton_rope_npu(..., num_stages: tl.constexpr):
    # cos/sin 在循环外加载
    for h_block in range(0, n_qh, BLOCK_H):
        # 编译器自动展开 num_stages 级流水线
        ...
```
- 通过 autotune 或 kernel 参数传入 `num_stages`；
- 保持循环内 load/compute/store 模式固定，便于后端做流水线；
- 与 head 分块结合使用效果更佳。

**常见坑**：
- `num_stages` 过大导致寄存器压力增加；
- 循环内有动态分支会破坏流水线；
- 不是强制参数，应通过 autotune 搜索。

### 6. `do_not_specialize`：避免 seq_len 变化导致编译风暴

**问题根因**：推理服务中 `seq_len` 随请求变化，Triton 默认会对每个不同值做启动级特化，产生大量 binary。

**收益明显条件**：
- 推理阶段 `seq_len` 动态；
- kernel 内 `seq_len` 仅用于 mask，不影响 tile shape；
- 编译缓存或启动延迟是瓶颈。

**实现要点**：
```python
@triton.jit(do_not_specialize=["seq_len"])
def _rope_forward_kernel(..., seq_len, ...):
    seq_mask = seq_offsets < seq_len
```
- 仅把 `seq_len` 加入 `do_not_specialize`；
- 用 mask 处理越界；
- 不要对 shape-defining 参数使用。

**常见坑**：
- 把 `BLOCK_T`、`hd`、`n_qh` 等常量参数也加入 `do_not_specialize`；
- 用 `seq_len` 做 `tl.cdiv` 或循环边界，导致无法动态；
- 忽略 mask 导致越界读写。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_rope_npu_001` | 35% | NPU 专用 RoPE 内存访问优化：同一位置 cos/sin 在 head 循环外加载并复用，结合软件流水线与 head 分块，支持前反向合并。 |
| `exp_rope_forward_001` | 25% | 通用 RoPE 前向内存访问优化：ALIGNED 路径一次性加载完整 rope_dim，非对齐路径分两半加载，并用 `do_not_specialize=["seq_len"]` 适配动态序列长度。 |

## 代表 Kernel

选择 `exp_rope_forward_001` 作为代表，因为它完整展示了 **cos/sin 复用、ALIGNED/非对齐路径、`do_not_specialize`** 三个内存访问核心模式。`exp_rope_npu_001` 可视为在此基础上增加 head 分块和软件流水线。

```python
@triton.jit(do_not_specialize=["seq_len"])
def _rope_forward_kernel(
    q_ptr,
    q_batch_stride,
    q_seq_stride,
    k_ptr,
    k_batch_stride,
    k_seq_stride,
    cos_ptr,
    cos_row_stride,
    sin_ptr,
    sin_row_stride,
    seq_len,
    num_seq_blocks,
    chunk_indices_ptr,
    kv_lens_ptr,
    bs: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_KV_LENS: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)
    total_blocks = bs * num_seq_blocks

    for block_id in range(pid, total_blocks, grid_size):
        if IS_VARLEN:
            chunk_idx = tl.load(chunk_indices_ptr + block_id * 5 + 1)
            seq_start = tl.load(chunk_indices_ptr + block_id * 5 + 2)
            sin_cos_offset = tl.load(chunk_indices_ptr + block_id * 5 + 3)
            actual_seq_len = tl.load(chunk_indices_ptr + block_id * 5 + 4)

            block_start_seq_idx = chunk_idx * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < actual_seq_len
            global_seq_offsets = seq_start + seq_offsets
            sin_cos_seq_offsets = sin_cos_offset + seq_offsets
            batch_idx = 0
        else:
            batch_idx = block_id // num_seq_blocks
            seq_block_id = block_id % num_seq_blocks

            block_start_seq_idx = seq_block_id * TOKEN_BLOCK_SIZE
            seq_offsets = block_start_seq_idx + tl.arange(0, TOKEN_BLOCK_SIZE)
            seq_mask = seq_offsets < seq_len
            global_seq_offsets = seq_offsets

            if HAS_KV_LENS:
                kv_len = tl.load(kv_lens_ptr + batch_idx)
                sin_cos_seq_offsets = kv_len + seq_offsets
            else:
                sin_cos_batch_offset = tl.where(cos_bs == 1, 0, batch_idx * seq_len)
                sin_cos_seq_offsets = sin_cos_batch_offset + seq_offsets

        half_rope_dim_offsets = tl.arange(0, half_rope_dim)
        half_rope_dim_mask = half_rope_dim_offsets < half_rope_dim

        # cos/sin 在 head 循环外加载一次，所有 head 复用
        cos_token_ptr = cos_ptr + sin_cos_seq_offsets[:, None] * cos_row_stride
        sin_token_ptr = sin_ptr + sin_cos_seq_offsets[:, None] * sin_row_stride
        cos_block_2d = tl.load(
            cos_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        sin_block_2d = tl.load(
            sin_token_ptr + half_rope_dim_offsets[None, :],
            mask=seq_mask[:, None] & half_rope_dim_mask[None, :],
            other=0,
        )
        cos_tile = tl.reshape(cos_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)
        sin_tile = tl.reshape(sin_block_2d, (TOKEN_BLOCK_SIZE, 1, half_rope_dim), can_reorder=True)

        head_q_offsets = tl.arange(0, n_qh)
        head_k_offsets = tl.arange(0, n_kh)

        if ALIGNED:
            # ALIGNED：单次加载完整 rope_dim
            rope_dim_offsets = tl.arange(0, rope_dim)
            rope_dim_mask = rope_dim_offsets < rope_dim

            q_offsets = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            q_mask = (
                seq_mask[:, None, None]
                & (head_q_offsets[None, :, None] < n_qh)
                & rope_dim_mask[None, None, :]
            )
            q_tile = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0).to(sin_block_2d.dtype)
            q_tile = _compute_rope(q_tile, sin_tile, cos_tile, n_qh, half_rope_dim, TOKEN_BLOCK_SIZE, False)
            tl.store(q_ptr + q_offsets, q_tile, mask=q_mask)

            k_offsets = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + rope_dim_offsets[None, None, :]
            )
            k_mask = (
                seq_mask[:, None, None]
                & (head_k_offsets[None, :, None] < n_kh)
                & rope_dim_mask[None, None, :]
            )
            k_tile = tl.load(k_ptr + k_offsets, mask=k_mask, other=0).to(sin_block_2d.dtype)
            k_tile = _compute_rope(k_tile, sin_tile, cos_tile, n_kh, half_rope_dim, TOKEN_BLOCK_SIZE, False)
            tl.store(k_ptr + k_offsets, k_tile, mask=k_mask)
        else:
            # 非 ALIGNED：分前后两半加载
            q_offsets_half1 = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_offsets_half2 = (
                batch_idx * q_batch_stride
                + global_seq_offsets[:, None, None] * q_seq_stride
                + head_q_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            q_half_mask = (
                seq_mask[:, None, None]
                & (head_q_offsets[None, :, None] < n_qh)
                & half_rope_dim_mask[None, None, :]
            )

            q_tile_1 = tl.load(q_ptr + q_offsets_half1, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            q_tile_2 = tl.load(q_ptr + q_offsets_half2, mask=q_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_q_1, new_q_2 = _compute_rope_separated(q_tile_1, q_tile_2, sin_tile, cos_tile, False)
            tl.store(q_ptr + q_offsets_half1, new_q_1, mask=q_half_mask)
            tl.store(q_ptr + q_offsets_half2, new_q_2, mask=q_half_mask)

            k_offsets_half1 = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_offsets_half2 = (
                batch_idx * k_batch_stride
                + global_seq_offsets[:, None, None] * k_seq_stride
                + head_k_offsets[None, :, None] * hd
                + nope_dim
                + half_rope_dim
                + half_rope_dim_offsets[None, None, :]
            )
            k_half_mask = (
                seq_mask[:, None, None]
                & (head_k_offsets[None, :, None] < n_kh)
                & half_rope_dim_mask[None, None, :]
            )

            k_tile_1 = tl.load(k_ptr + k_offsets_half1, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            k_tile_2 = tl.load(k_ptr + k_offsets_half2, mask=k_half_mask, other=0.0).to(sin_block_2d.dtype)
            new_k_1, new_k_2 = _compute_rope_separated(k_tile_1, k_tile_2, sin_tile, cos_tile, False)
            tl.store(k_ptr + k_offsets_half1, new_k_1, mask=k_half_mask)
            tl.store(k_ptr + k_offsets_half2, new_k_2, mask=k_half_mask)
```

**关键实现细节**：
- `do_not_specialize=["seq_len"]` 避免推理长度变化时重复编译；
- cos/sin 在 head 循环外加载一次，reshape 后广播到所有 head；
- `ALIGNED` 路径用单次 load/store 处理完整 `rope_dim`；
- 非 `ALIGNED` 路径把 `rope_dim` 拆成两半，分别加载、旋转、写回；
- `IS_VARLEN`/`HAS_KV_LENS` 用 `tl.constexpr` 编译期分支，无运行时开销。

## 已知实例

- `exp_rope_npu_001` — Memory Access（收益 35%）
  - NPU 专用 RoPE 内存访问优化：cos/sin 复用、软件流水线、head 分块、前反向合并。对应 kernel: `raw_experiences/extracted_kernels/104-_triton_rope_npu.py`。
- `exp_rope_forward_001` — Memory Access（收益 25%）
  - 通用 RoPE 前向内存访问优化：ALIGNED 路径、非对齐分半加载、`do_not_specialize=["seq_len"]`。对应 kernel: `raw_experiences/extracted_kernels/229-_rope_forward_kernel.py`。

## 关联知识

- 开发指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md)

<!-- okf:related:start -->

# 相关

- 对应指南: [融合算子综合页](../../reference/op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, rope
- 对应指南: [RoPE（旋转位置编码）算子优化](../../reference/op-class-rope.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../../ops/mla-preprocess.md) — 主题相关：共享 rope
- 相关主题: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../../ops/mla-prolog-v3.md) — 主题相关：共享 rope
- 相关主题: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../../ops/mla-prolog.md) — 主题相关：共享 rope
- 相关主题: [RMSNorm-RoPE fused kernel on Ascend 910B1](../../ops/rmsnorm-rope.md) — 主题相关：共享 rope
- 相关主题: [RoPE paired-channel pointer addressing](../../ops/rope-pointer-addressing.md) — 主题相关：共享 rope
- 相关主题: [FlashAttention 变体的内存访问优化](../algorithm-numerics/traj-attention-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [用 tl.constexpr 编译期消除运行时分支](../compute-pipeline/constexpr-branch-elimination.md) — 同属优化知识，主题相关
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rope
- 相关主题: [Causal Conv1D 内存访问优化：滑动窗口寄存器复用、Block Pointer 与重叠 Tiling](../compute-pipeline/traj-conv-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](../compute-pipeline/traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, cos_sin_reuse, rope
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [动态序列长度场景使用 do_not_specialize 降低编译开销](../meta-skills/traj-misc-compilation_opt.md) — 主题相关：共享 910b2, 910b3, do_not_specialize
- 相关主题: [Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局](../portability/traj-rope-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [输入进入 kernel 前必须 contiguous](input-contiguous.md) — 主题相关：共享 memory_access
- 相关主题: [Activation Backward 内存访问优化聚合](traj-activation-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Embedding 与索引拷贝内存访问优化：Block Pointer、变长 cumsum 与右 Padding 偏移](traj-embedding-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Gather/Scatter 内存访问优化聚合](traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Ascend 大 Shape 连续分块内存访问优化](traj-general_ascend-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [通用内存访问优化：连续化、Block Pointer、寄存器复用与标量边界消除](traj-general_perf-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Matmul 内存访问优化：分块策略、L0 复用与算子融合](traj-matmul-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Misc 算子内存访问优化聚合](traj-misc-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Sum-and-Scale 归约内存访问优化](traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Softmax Backward 内存访问优化聚合](traj-softmax-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [Solve_Tril 内存访问优化：Preload + extract_slice 消除循环内 HBM 重读](traj-solve_tril-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access
- 相关主题: [SSM 内存访问优化：转置 load 合并化、寄存器复用与分块状态矩阵](traj-ssm-memory_access.md) — 主题相关：共享 910b2, 910b3, memory_access

<!-- okf:related:end -->
