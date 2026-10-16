---
schema_version: okf.v1
kind: example
type: code_example
source_family: trajectory
category: rope
dimension: "4.1"
title: Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局
description: 针对 Ascend NPU 内存层次结构优化的 Llama4 RoPE 实现，覆盖 UB tiling 策略、tl.math.fma 融合乘加指令利用，以及交错复数布局（real/imag）优化，收益 25%。
tags: [rope, hardware_adaptation, llama4, ascend_npu, ub_tiling, fma, interleaved_layout, 910b2, 910b3]
confidence: verified
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/raw/experiences/rope/
created_at: "2026-08-11T00:00:00Z"
updated_at: "2026-08-12T00:00:00Z"
---

# Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB Tiling、FMA 与交错复数布局

## 概述

Llama4 的 RoPE 采用**交错复数布局**（interleaved complex layout）：head 维中相邻两个元素分别代表实部和虚部，旋转操作等价于复数乘法 `(a+bi)(c+di)`。在 Ascend 910B NPU 上，要高效实现这一模式需要适配三层硬件特性：
1. **UB（Unified Buffer）容量有限**：必须对 Q/K head 做分块，让活跃工作集留在 UB；
2. **FMA（Fused Multiply-Add）指令**：复数乘法可拆解为两条 FMA，比独立乘加指令数更少；
3. **Vector 单元偏好交错布局**：用 `tl.interleave` / `tl.split` 配合 `can_reorder=True`，避免显式转置或 gather/scatter。

本卡唯一的 experience `exp_rope_llama4_npu_001` 正是围绕这三点进行硬件适配，收益 25%。

**核心 trade-off**：
- UB tiling 要求 BLOCK_Q/BLOCK_K 足够小以驻留 UB，但过小会增加循环次数；
- FMA 需要把复数乘法整理成 `fma(x, y, z)` 形式，代码比直接 `*`/`+` 略复杂；
- 交错布局优化依赖编译器 reorder，若后端不支持 `can_reorder=True` 需回退到显式 reshape。

## 共同模式

### 1. UB Tiling：按 Q/K head 分块

**问题根因**：Ascend NPU 的 UB 容量远小于 HBM。如果一次性加载某个位置的所有 Q/K head，tile 尺寸会超过 UB，导致频繁换入换出。

**收益明显条件**：
- `n_qh`、`n_kh` 较大（如 ≥ 32/64）；
- `hd` 能被 2 整除，支持复数对处理；
- 每个位置需要处理多个 head。

**实现要点**：
```python
for qh_block in range(0, n_qh, BLOCK_Q):
    qh_idx = tl.arange(0, BLOCK_Q) + qh_block
    qh_mask = qh_idx < n_qh
    block_mask = qh_mask[:, None] & hd_mask[None, :]
    head_ptr = q_base + qh_idx[:, None] * q_head_stride
    q_pair = tl.load(head_ptr + hd_idx[None, :], mask=block_mask, other=0.0)
    # ... 处理并写回 ...
```
- 外层按 head 分块，内层处理一个 BLOCK_Q/BLOCK_K 的 head 集合；
- 同一位置的 freqs_complex 在 head 间复用；
- 保证 block 工作集能放入 UB。

**常见坑**：
- BLOCK_Q/BLOCK_K 太大导致 UB spill；
- 太小导致循环开销和重复加载 freqs；
- head 维 stride 计算错误会导致数据错位。

### 2. FMA 指令：复数乘法的硬件友好形式

**问题根因**：复数乘法 `(a+bi)(c+di) = (ac-bd) + (ad+bc)i` 需要 4 次乘法和 2 次加减。直接用 `*` 和 `+` 会生成 6 条独立指令，而 FMA 可以合并为 2 条核心运算。

**收益明显条件**：
- RoPE 旋转是核心热点；
- 硬件支持 fused multiply-add（Ascend Vector 单元支持）；
- 输入已经是实部/虚部分离的张量。

**实现要点**：
```python
new_real = tl.math.fma(q_real, freqs_real, -(q_imag * freqs_imag))
new_imag = tl.math.fma(q_real, freqs_imag,  q_imag * freqs_real)
```
- 先 split 出 `real` 和 `imag`；
- 用 `tl.math.fma(x, y, z)` 计算 `x*y + z`；
- 注意符号：实部中 `-bd` 通过 `-(q_imag * freqs_imag)` 作为 FMA 的第三个操作数。

**常见坑**：
- 不要把 FMA 和独立乘加混用，否则失去融合优势；
- 符号处理错误会改变旋转方向；
- 低精度场景下 FMA 中间结果精度需确认。

### 3. 交错复数布局优化

**问题根因**：Llama4 的频率和 Q/K 数据按 `(..., hd//2, 2)` 或等效交错形式存储。若用显式 transpose/gather 拆出实部和虚部，会引入大量非合并访问和标量索引。

**收益明显条件**：
- 输入/输出是交错复数布局；
- 后端支持 `can_reorder=True` 和 `tl.interleave`/`tl.split`；
- 需要在 real/imag 之间快速转换。

**实现要点**：
```python
freqs_complex = tl.load(freqs_complex_ptr + freq_base + freq_idx, mask=freq_mask, other=0.0)
freqs_complex = freqs_complex.reshape(hd // 2, 2, can_reorder=True)
freqs_real, freqs_imag = tl.split(freqs_complex)
freqs_imag = freqs_imag * imag_sign

q_pair = q_pair.reshape(BLOCK_Q, hd // 2, 2, can_reorder=True)
q_real, q_imag = tl.split(q_pair)
# ... FMA 计算 ...
new_q_pair = tl.interleave(new_real, new_imag)
tl.store(head_ptr + hd_idx[None, :], new_q_pair, mask=block_mask)
```
- `reshape(..., can_reorder=True)` 让编译器选择最优内存布局；
- `tl.split` 把最后一维 2 拆成 real/imag；
- `tl.interleave` 把 real/imag 重新合并成交错布局。

**常见坑**：
- `can_reorder=True` 在不同后端行为可能不同，需在 Ascend 上验证；
- `reshape` 后的维度顺序要与 `split/interleave` 匹配；
- 输出必须用 `tl.interleave` 恢复原始交错布局，否则下游读取错误。

### 4. 频率复用与边界处理

**问题根因**：同一 seq 位置的所有 head 共享同一组 freqs_complex。如果每个 head block 都重复加载频率，会浪费 UB 带宽。

**收益明显条件**：
- `n_qh`、`n_kh` 多，freqs 复用价值高；
- freqs 张量不大，可常驻 UB。

**实现要点**：
- 在 Q/K head 循环外加载 `freqs_complex`；
- split 成 `freqs_real`/`freqs_imag` 后供所有 head block 复用；
- 用 mask 处理 `batch_idx >= bs` 等边界。

**常见坑**：
- freqs 加载放在 head 循环内会导致重复读取；
- `batch_idx >= bs` 的 program 要提前 return，避免无效计算；
- 频率的 `imag_sign` 在 forward/backward 中可能相反，需用 `tl.constexpr` 特化。

## 收益统计

| experience | 收益 | 简介 |
|---|---|---|
| `exp_rope_llama4_npu_001` | 25% | Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB head 分块、tl.math.fma 复数乘法、交错复数布局的 reshape/split/interleave 优化。 |

## 代表 Kernel

选择唯一的 `exp_rope_llama4_npu_001` 作为代表。以下为该 kernel 的代码，展示了 **UB tiling、FMA、交错复数布局** 三个核心模式：

```python
@triton.jit
def _triton_llama4_rope_npu(
    q_ptr,
    k_ptr,
    freqs_complex_ptr,
    q_row_stride,
    k_row_stride,
    q_head_stride,
    k_head_stride,
    freqs_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    imag_sign: tl.constexpr,
):
    """
    Llama4 RoPE on Ascend NPU for interleaved complex layout:
    - q/k shape: (bs, sl, n_heads, hd)
    - freqs_complex_ptr: (sl, hd//2, 2)
    """
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // sl
    seq_idx = pid % sl

    if batch_idx >= bs:
        return

    q_base = q_ptr + pid * q_row_stride
    k_base = k_ptr + pid * k_row_stride

    freq_base = seq_idx * freqs_row_stride
    hd_idx = tl.arange(0, hd)
    hd_mask = hd_idx < hd

    freq_idx = tl.arange(0, hd)
    freq_mask = freq_idx < hd

    # 同一位置所有 head 复用同一组 freqs
    freqs_complex = tl.load(
        freqs_complex_ptr + freq_base + freq_idx,
        mask=freq_mask, other=0.0,
    )
    freqs_complex = freqs_complex.reshape(hd // 2, 2, can_reorder=True)
    freqs_real, freqs_imag = tl.split(freqs_complex)
    freqs_imag = freqs_imag * imag_sign

    # Q heads：按 BLOCK_Q 分块，适配 UB
    for qh_block in range(0, n_qh, BLOCK_Q):
        qh_idx = tl.arange(0, BLOCK_Q) + qh_block
        qh_mask = qh_idx < n_qh
        block_mask = qh_mask[:, None] & hd_mask[None, :]

        head_ptr = q_base + qh_idx[:, None] * q_head_stride
        q_pair = tl.load(
            head_ptr + hd_idx[None, :],
            mask=block_mask,
            other=0.0,
        )
        q_pair = q_pair.reshape(BLOCK_Q, hd // 2, 2, can_reorder=True)
        q_real, q_imag = tl.split(q_pair)

        # FMA 计算复数乘法
        new_real = tl.math.fma(q_real, freqs_real, -(q_imag * freqs_imag))
        new_imag = tl.math.fma(q_real, freqs_imag, q_imag * freqs_real)
        new_q_pair = tl.interleave(new_real, new_imag)

        tl.store(head_ptr + hd_idx[None, :], new_q_pair, mask=block_mask)

    # K heads：按 BLOCK_K 分块
    for kh_block in range(0, n_kh, BLOCK_K):
        kh_idx = tl.arange(0, BLOCK_K) + kh_block
        kh_mask = kh_idx < n_kh
        block_mask = kh_mask[:, None] & hd_mask[None, :]

        head_ptr = k_base + kh_idx[:, None] * k_head_stride
        k_pair = tl.load(
            head_ptr + hd_idx[None, :],
            mask=block_mask,
            other=0.0,
        )
        k_pair = k_pair.reshape(BLOCK_K, hd // 2, 2, can_reorder=True)
        k_real, k_imag = tl.split(k_pair)

        new_real = tl.math.fma(k_real, freqs_real, -(k_imag * freqs_imag))
        new_imag = tl.math.fma(k_real, freqs_imag, k_imag * freqs_real)
        new_k_pair = tl.interleave(new_real, new_imag)

        tl.store(head_ptr + hd_idx[None, :], new_k_pair, mask=block_mask)
```

**关键实现细节**：
- 按 `BLOCK_Q/BLOCK_K` 对 head 分块，确保每块工作集能放入 UB；
- 同一 `seq_idx` 的 `freqs_complex` 在 head 循环外加载一次并 split，供所有 head 复用；
- 复数乘法用 `tl.math.fma` 实现，减少独立指令数；
- 输入/输出保持交错复数布局，通过 `reshape(..., can_reorder=True)`、`tl.split`、`tl.interleave` 在寄存器内转换，避免显式 transpose；
- `imag_sign` 用 `tl.constexpr` 特化 forward/backward 符号。

## 已知实例

- `exp_rope_llama4_npu_001` — Hardware Adaptation（收益 25%）
  - Llama4 RoPE 在 Ascend NPU 上的硬件适配：UB head 分块、`tl.math.fma` 复数乘法、交错复数布局的 reshape/split/interleave 优化。代表 kernel 见上。

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
- 相关主题: [算子融合模式与约束决策](../compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 rope
- 相关主题: [RoPE 计算优化：cos/sin 复用、向量化旋转与多模态位置编码融合](../compute-pipeline/traj-rope-compute.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [RoPE 反向调度优化：Grid-Stride 循环、变长序列与 KV Cache 索引](../compute-pipeline/traj-rope-scheduling.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [RoPE 内存访问优化：cos/sin 复用、ALIGNED 向量化加载与动态特化](../memory-sram/traj-rope-memory_access.md) — 主题相关：共享 910b2, 910b3, rope
- 相关主题: [Attention 硬件适配优化聚合](traj-attention-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Ascend 硬件资源适配规则聚合](traj-general_ascend-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [Entropy Backward 在 Ascend NPU 上的硬件适配优化](traj-loss-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend_npu, hardware_adaptation
- 相关主题: [Matmul 硬件适配优化聚合](traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, hardware_adaptation
- 相关主题: [SDA（Scaled Dot Attention）Ascend 硬件适配：HiVM 分块、多缓冲预取与 Cube/Vector 均衡](traj-misc-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, ascend_npu, hardware_adaptation

<!-- okf:related:end -->
