---
title: 归一化算子优化
category: ops
tags: [normalization, layernorm, rmsnorm, softmax, online-algorithm, numerical-stability]
hardware: [910b2, 910b3]
last_updated: 2026-04-27
---

# 归一化算子优化

## 概述

归一化算子对张量的某一维度（通常是隐藏维度）计算统计量（均值、方差、RMS），然后对输入进行标准化变换，并可选地应用仿射变换（weight × norm + bias）。典型算子包括 LayerNorm、RMSNorm、Softmax 以及它们的融合变体。

在 Ascend 910b2/910b3 上，归一化算子优化的核心挑战在于：

1. **数值精度**：fp16/bf16 累加易溢出，需要混合精度策略
2. **内存容量**：UB（Unified Buffer）约 192KB，超大隐藏维度需要分块处理
3. **访存效率**：两遍扫描（two-pass）可以通过在线算法压缩为一遍，减少 HBM 读取次数
4. **融合机会**：skip-add、残差更新、量化等操作与归一化高度可融合

## 检测标准

满足以下任意模式时，将算子归类为 normalization：

- `mean` / `var` / `rms` 后跟 `(x - mean) / std` 或 `x / rms` 的标准化变换
- `weight * norm + bias` 仿射变换（与标准化紧密耦合）
- `rsqrt(var + eps)` 出现在计算图中
- 独立的 `softmax`（沿某维度的 exp + sum_reduce + div）
- PyTorch 原语：`F.layer_norm`、`F.rms_norm`、`F.softmax`、`torch.nn.LayerNorm`

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | 输入 fp16/bf16；累加必须 fp32；输出可 fp16/bf16/int8（量化） |
| 隐藏维度 n_cols | n_cols ≤ UB 容量（~192KB / dtype_bytes）时可单 tile 处理；否则需列分块 |
| 批量维度 n_rows | 通常每个 program 处理一行，grid=(n_rows,) |
| weight/bias | 可缓存在寄存器中（当 n_cols 较小时），避免重复从 HBM 加载 |
| 与残差结合 | skip-add + norm 可融合为单内核，消除中间张量 |

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 内核参考 |
|------|--------|-------|----------|----------|
| RMSNorm 基础优化 | exp_rmsnorm_001 | 25% | fp32 混合精度累加，weight 缓存寄存器 | kernel 045, 051 |
| RMSNorm 大批量 | exp_rmsnorm_001 | 35% | Persistent kernel，grid=num_aicores，weight 仅加载一次 | kernel 086 |
| Skip + LayerNorm 融合 | exp_skip_layernorm_001 | 30% | 跳跃连接 add 与 LayerNorm 融合，两遍在寄存器内完成 | kernel 118 |
| 残差原位更新 + RMSNorm | exp_fused_rms_norm_001 | 35% | 计算方差时同步完成残差原位更新 | kernel 115 |
| 超大隐藏维度 RMSNorm | exp_rmsnorm_npu_001 | 35% | UB 列分块（BLOCK_SIZE_N），两遍遍历 | kernel 093 |
| RMSNorm + 量化 | exp_rmsnorm_fused_quant_001 | 50% | 归一化与量化在单内核融合 | kernel 189 |
| LayerNorm 条件分支 | exp_layernorm_fwd_001 | 30% | constexpr HAS_BIAS/HAS_RESIDUAL 编译期分支消除 | kernel 150 |
| NPU SIMD + LayerNorm + 门控 | exp_layernorm_npu_simd_001 | 35% | NPU SIMD 指令 + LayerNorm + gating 融合 | kernel 180 |
| Gemma 风格 RMSNorm | exp_gemma_rmsnorm_001 | 35% | (1 + weight) × norm 融合计算 | kernel 190 |
| Q/K 双路 RMSNorm | exp_fused_split_qk_norm_001 | 35% | Q 和 K 的 RMSNorm 在单内核内并行完成 | kernel 191 |
| Softmax 单遍 | exp_online_softmax_001 | 35% | log-sum-exp trick 单遍 softmax | — |
| 均值+方差单遍 | exp_online_mean_var_001 | 30% | Welford 在线算法，均值和方差一遍完成 | — |
| RMS 单遍 | exp_online_rms_001 | 25% | 单遍 RMS 计算 | — |
| LayerNorm Welford | exp_layernorm_single_pass_001 | 30% | Welford 用于 LayerNorm，单遍输入扫描 | kernel 181 |

## 关键技术

### 混合精度累加

fp16/bf16 的动态范围不足以安全累加大量平方项（`sum(x²)`），必须在 fp32 中累加：

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_row_ptr = x_ptr + row * n_cols

    # Pass 1: 以 fp32 累加 sum(x^2)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)          # 提升到 fp32
        acc += x_fp32 * x_fp32

    var = tl.sum(acc) / n_cols
    rrms = tl.rsqrt(var + eps)             # 硬件 rsqrt 指令

    # Pass 2: 归一化 + 仿射变换
    # （weight 可在寄存器中缓存，若 n_cols 足够小）
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = (x * rrms) * w
        tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)
```

### rsqrt 替代 1/sqrt

Ascend 提供硬件 `rsqrt` 指令，比 `1 / sqrt(x)` 快显著：

```python
# 低效
inv_std = 1.0 / tl.sqrt(var + eps)

# 高效：使用硬件 rsqrt 指令
inv_std = tl.rsqrt(var + eps)
```

### UB 列分块（Column Blocking）

当 `n_cols × dtype_bytes > ~192KB` 时，单 tile 无法放入 UB，必须分块两遍扫描：

```python
@triton.jit
def rmsnorm_large_hidden_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps,
                                 BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols

    # Pass 1: 分块计算 sum(x^2)
    var_acc = tl.zeros([1], dtype=tl.float32)
    for col_start in range(0, n_cols, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        var_acc += tl.sum(x * x)
    var = var_acc / n_cols
    rrms = tl.rsqrt(var + eps)

    # Pass 2: 分块归一化
    for col_start in range(0, n_cols, BLOCK_SIZE_N):
        cols = col_start + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = x * rrms * w
        tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)
```

**BLOCK_SIZE_N 推荐值**（910b2，UB ~192KB）：

| 数据类型 | 最大 BLOCK_SIZE_N |
|----------|------------------|
| fp16 / bf16 (2B) | ~49152 元素 |
| fp32 (4B) | ~24576 元素 |

### Online Softmax（单遍 log-sum-exp）

传统 softmax 需要两遍（先求 max，再求 exp+sum），在线算法可在单遍完成：

```python
@triton.jit
def online_softmax_kernel(x_ptr, y_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols

    # 单遍：同时维护 running max 和 running sum
    running_max = tl.full([1], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([1], dtype=tl.float32)

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        block_max = tl.max(x)
        new_max = tl.maximum(running_max, block_max)
        # 更新 running_sum 时补偿 max 变化
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      tl.sum(tl.exp(x - new_max))
        running_max = new_max

    # 写回归一化结果
    log_sum = tl.log(running_sum) + running_max   # log-sum-exp
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.exp(x - log_sum)
        tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)
```

### Welford 在线均值/方差

Welford 算法单遍稳定计算均值和方差，避免数值不稳定：

```python
# Welford 在线更新（在 BLOCK_SIZE 维度上应用）
mean = tl.zeros([1], dtype=tl.float32)
m2 = tl.zeros([1], dtype=tl.float32)
count = tl.zeros([1], dtype=tl.float32)

for col_start in range(0, n_cols, BLOCK_SIZE):
    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    block_count = tl.sum(mask.to(tl.float32))
    block_mean = tl.sum(x * mask) / block_count
    # Welford 合并
    delta = block_mean - mean
    count += block_count
    mean += delta * block_count / count
    m2 += tl.sum((x - mean) * (x - block_mean) * mask)

var = m2 / count
```

### Persistent Kernel + Weight 缓存

对 n_rows 很大的情况，持久化内核使 weight 仅加载一次：

```python
@triton.jit
def persistent_rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_rows, n_cols, eps,
                               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # weight 在寄存器中缓存（每个 program 只加载一次）
    cols = tl.arange(0, BLOCK_SIZE)
    mask_w = cols < n_cols
    w = tl.load(w_ptr + cols, mask=mask_w, other=1.0).to(tl.float32)

    # grid-stride 循环处理所有行
    for row in range(pid, n_rows, num_progs):
        x_row = x_ptr + row * n_cols
        # ... 计算 RMSNorm ...
        var = tl.sum(x_fp32 * x_fp32) / n_cols
        rrms = tl.rsqrt(var + eps)
        y = x_fp32 * rrms * w
        tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask_w)

# 启动：grid 固定为 AI Core 数量
grid = (NUM_AI_CORES,)
```

### constexpr 标志消除编译期分支

使用 `tl.constexpr` 标志在编译期消除 HAS_BIAS、HAS_RESIDUAL 等条件分支，避免运行时 branch 开销：

```python
@triton.jit
def layernorm_kernel(x_ptr, residual_ptr, w_ptr, b_ptr, y_ptr,
                     n_cols, eps,
                     HAS_BIAS: tl.constexpr,
                     HAS_RESIDUAL: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask).to(tl.float32)

    if HAS_RESIDUAL:  # 编译期展开，无运行时开销
        res = tl.load(residual_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
        x = x + res

    mean = tl.sum(x) / n_cols
    var = tl.sum((x - mean) * (x - mean)) / n_cols
    x_norm = (x - mean) * tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    y = x_norm * w

    if HAS_BIAS:      # 编译期展开
        b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
        y = y + b

    tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)
```

### RMSNorm + 量化融合

归一化结果直接量化为 int8，消除中间 fp16 张量的写回（50% 提升）：

```python
@triton.jit
def rmsnorm_quant_fused_kernel(x_ptr, w_ptr, scale_ptr, y_int8_ptr,
                                n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)

    var = tl.sum(x * x) / n_cols
    rrms = tl.rsqrt(var + eps)
    y_norm = x * rrms * w

    # 直接量化为 int8，无中间 HBM 写回
    scale = tl.load(scale_ptr)
    y_int8 = tl.clamp(tl.round(y_norm / scale), -128, 127).to(tl.int8)
    tl.store(y_int8_ptr + row * n_cols + cols, y_int8, mask=mask)
```

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| fp16 累加溢出 | `sum(x²)` 在 fp16 下对大 n_cols 易溢出 | 始终在 fp32 下累加（exp_rmsnorm_001） |
| UB 溢出 | n_cols > UB 容量时编译失败或性能骤降 | 使用列分块（exp_rmsnorm_npu_001），BLOCK_SIZE_N ≤ 49152（fp16） |
| 两遍 HBM 读取 | 传统两遍算法读两次输入，带宽受限 | 使用 Welford / 在线算法单遍完成（exp_online_*） |
| weight 重复加载 | 每行都从 HBM 加载 weight，造成大量冗余访问 | Persistent kernel 将 weight 缓存在寄存器（exp_rmsnorm_001 kernel 086） |
| softmax 数值不稳定 | 未减 max 直接 exp 导致溢出 | 使用 log-sum-exp trick（exp_online_softmax_001） |
| 独立的 skip-add 内核 | add + norm 分开执行，多一次 HBM 往返 | 融合 skip-add + LayerNorm（exp_skip_layernorm_001） |
| 忽略 eps 的精度影响 | 极小 var 时 rsqrt 精度不足 | eps 使用 fp32 精度，不随输入 dtype 降精度 |

## 代码模板

### 标准 RMSNorm 模板

```python
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols
    y_row = y_ptr + row * n_cols

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # 加载并提升到 fp32
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)

    # 计算 RMS
    var = tl.sum(x * x) / n_cols
    rrms = tl.rsqrt(var + eps)

    # 归一化 + 仿射
    y = x * rrms * w

    tl.store(y_row + cols, y.to(tl.float16), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.contiguous()
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    # BLOCK_SIZE 需满足 BLOCK_SIZE >= n_cols 且为 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    assert BLOCK_SIZE <= 65536, f"n_cols={n_cols} 超出单 tile UB 容量，需列分块"
    rmsnorm_kernel[(n_rows,)](x, weight, y, n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y
```

### Skip + LayerNorm 融合模板

```python
@triton.jit
def skip_layernorm_kernel(
    x_ptr, residual_ptr, w_ptr, b_ptr,
    y_ptr, residual_out_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask).to(tl.float32)
    res = tl.load(residual_ptr + row * n_cols + cols, mask=mask).to(tl.float32)

    # skip add（在寄存器内完成）
    x = x + res

    # 可选：写回残差（供下一层使用）
    tl.store(residual_out_ptr + row * n_cols + cols, x.to(tl.float16), mask=mask)

    # LayerNorm
    mean = tl.sum(x) / n_cols
    var = tl.sum((x - mean) * (x - mean)) / n_cols
    x_norm = (x - mean) * tl.rsqrt(var + eps)

    w = tl.load(w_ptr + cols, mask=mask).to(tl.float32)
    y = x_norm * w
    if HAS_BIAS:
        b = tl.load(b_ptr + cols, mask=mask).to(tl.float32)
        y = y + b

    tl.store(y_ptr + row * n_cols + cols, y.to(tl.float16), mask=mask)
```

## 相关页面

- [elementwise.md](./elementwise.md) — 归一化中的仿射变换（weight × norm + bias）本质是 elementwise 操作
- [reduction.md](./reduction.md) — 归一化内部的 sum、max 等归约原语
- [gather_scatter.md](./gather_scatter.md) — MoE 场景中，Expert 归一化（Q/K RMSNorm）与 gather/scatter 结合

## 反向传播经验

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| RMSNorm 反向 | exp_rmsnorm_backward_001 | 基础 RMSNorm dX/dW 计算 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_backward_001.yaml) |
| RMSNorm 反向（NPU） | exp_rmsnorm_backward_npu_001 | Ascend 910b 专用反向路径 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_backward_npu_001.yaml) |
| RMSNorm 反向（分块） | exp_rmsnorm_backward_tiled_001 | 大列维度分块反向 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_backward_tiled_001.yaml) |
| RMSNorm 反向（大列） | exp_rmsnorm_bwd_large_cols_001 | 超 UB 列维度的反向 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_bwd_large_cols_001.yaml) |
| RMSNorm bwd 基础 | exp_rmsnorm_bwd_001 | RMSNorm bwd 标准实现 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_bwd_001.yaml) |
| RMSNorm bwd 融合 | exp_rmsnorm_bwd_fused_001 | RMSNorm bwd 与其他算子融合 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_bwd_fused_001.yaml) |
| LayerNorm 反向 | exp_layernorm_bwd_001 | LayerNorm dX/dW/dB | [yaml](../../raw/experiences/normalization/exp_layernorm_bwd_001.yaml) |
| LayerNorm 反向融合 | exp_layernorm_bwd_fused_001 | LayerNorm bwd + residual 融合 | [yaml](../../raw/experiences/normalization/exp_layernorm_bwd_fused_001.yaml) |
| GroupNorm 反向 | exp_gn_backward_001 | GroupNorm dX/dW/dB | [yaml](../../raw/experiences/normalization/exp_gn_backward_001.yaml) |

## 推理与变体经验

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| RMSNorm 推理优化 | exp_rmsnorm_infer_001 | 推理专用路径，跳过训练相关计算 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_infer_001.yaml) |
| RMSNorm 无 Tiling | exp_rmsnorm_notiling_001 | 小列维度直接计算，无需分块 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_notiling_001.yaml) |
| RMSNorm KV 归一化 | exp_rmsnorm_kv_001 | KV cache 归一化专项优化 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_kv_001.yaml) |
| RMSNorm + Add | exp_rmsnorm_add_001 | RMSNorm 与 residual add 融合 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_add_001.yaml) |
| RMSNorm + RoPE | exp_rmsnorm_rope_001 | RMSNorm 与 RoPE 顺序融合 | [yaml](../../raw/experiences/normalization/exp_rmsnorm_rope_001.yaml) |

