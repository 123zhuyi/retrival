---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: MatMul / GEMM 高性能算子
description: MatMul / GEMM 高性能算子
tags: [matmul, gemm, tiling, cube, npu, 910b2, 910b3, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/matmul.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/matmul.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/matmul.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# MatMul / GEMM 高性能算子

## 概述

矩阵乘法（General Matrix Multiply，GEMM）是深度学习中最核心的计算原语，覆盖全连接层、注意力投影、MoE 专家路由等场景。在 Ascend 910b 系列硬件上，GEMM 的计算主要由 Cube 单元承担，Vector 单元负责后处理（激活、量化、偏置加法等）。通过合理的 Tiling、Core 映射与内存层次利用，Triton-Ascend GEMM 算子相比 PyTorch 基线可获得 10%–40% 的性能提升。

本页汇总了所有已验证的 GEMM 经验，涵盖基础 Tiling 策略、内存访问优化、多核调度、融合变体以及硬件特定技巧，供 Triton-Ascend 算子开发者查阅和复用。

## 检测标准

满足以下任意一条，认定为 GEMM 类算子：

- 计算图包含 `torch.matmul` / `torch.mm` / `torch.bmm` / `F.linear` 调用
- 输出形状满足 `[..., M, N]`，输入形状满足 `[..., M, K]` 和 `[..., K, N]`
- einsum 表达式含 `ij,jk->ik` / `bij,bjk->bik` 等矩阵收缩模式
- 算子包含可选的转置标志 `trans_a` / `trans_b`

## 适用条件

| 维度 | 最小值 | 推荐值 | 说明 |
|------|--------|--------|------|
| M    | 1      | ≥128   | 小于 128 走 micro-tiling 路径 |
| N    | 1      | ≥128   | 同上 |
| K    | 1      | ≥256   | K < 256 收益有限 |
| dtype | —     | fp16 / bf16 | float32 也支持，但 Cube 利用率偏低 |
| layout | —   | ND (行主序) | NZ layout 需额外转换，但 K 展开收益更高 |
| trans_a | — | False  | 双转置有专项优化 |
| trans_b | — | True   | 常见的 weight 转置 |

## 经验路由

### 基础经验

| 场景 | 经验 ID | 提升 | 核心技术 | 适用条件 |
|------|---------|------|----------|----------|
| 大矩阵标准 GEMM | exp_matmul_001 | 40% | L0级 128×128×128 Tiling，L0C 寄存器复用 | M≥256, N≥256, K≥1024, fp16/bf16, ND |
| 量化融合 | exp_matmul_002 | 35% | Cube fixpipe 量化 + 4× K 循环展开 + 预取 | M≥128, N≥128, K≥128, fp16, NZ |
| 行宽对齐 (A/B 不转置) | exp_matmul_003 | 10% | 严格 512B 行宽倍数对齐块划分 | M≥128, N≥256, K≥256, fp16/bf16/float, ND |
| K 宽统一分块 (B 转置) | exp_matmul_004 | 10% | K 轴统一分块 | M≥128, N≥128, K≥256, fp16/bf16/float, ND, trans_b |
| 双转置优化 | exp_matmul_005 | 32% | M=256, K=256, N=128 配置；双转置专用路径 | M≥256, N≥128, K≥256, fp16/bf16/float, ND, trans_a & trans_b |
| 小矩阵 | exp_matmul_micro_tiling_001 | 20% | Micro-tiling 16×16×16 | 任意维度 < 128 |

### 高性能 Tiling 技巧

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 列优先 Core 映射 | skill_tiling_default_col | — | GROUP_SIZE∈{8,16,32}，N 维度频繁访问 |
| 行优先 Core 映射 | skill_tiling_default_row | — | M 维度频繁访问 |
| 对角映射消除 bank conflict | skill_tiling_diagonal / exp_matmul_diag_tiling_001 | 25% | M≥512, N≥512；对角线遍历 |
| K 轴 offset 迭代 | skill_k_offset | — | K 维度极大场景 |
| AutoTiling 搜索 | exp_matmul_autotune_001 / autotiling_enum | — | BLOCK_M∈{128,256}, BLOCK_N∈{128,256}, BLOCK_K=256, GROUP_SIZE∈{8,16,32} |

### 内存访问优化

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| L0C 寄存器复用 | exp_matmul_mem_001 | 30% | 中间结果存 L0C，避免重复加载 |
| NZ 格式 + K 展开 | exp_matmul_mem_002 | 35% | NZ layout + 4× K 循环展开 |
| 循环不变量提升 (LICM) | exp_matmul_mem_003 | 20% | 外积外层循环不变量提升 |
| 2D Block Pointer | exp_matmul_mem_004 | 25% | `tl.make_block_ptr` 简化 2D 指针运算 |
| 混合精度分块 | exp_matmul_mem_005 | 30% | 分块矩阵乘 + 混合精度 |
| PID 分组 L2 局部性 | exp_matmul_mem_006 | 25% | 相邻 PID 访问同列/行，提升 L2 命中率 |

### 调度与多核优化

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 对角线块遍历 | exp_matmul_sched_001 | 25% | 对角遍历消除 bank conflict，M≥512, N≥512 |
| 多核对角划分 | exp_matmul_sched_002 | 35% | 24 核对角划分均衡负载 |
| Persistent Kernel | exp_matmul_sched_003 | 25–35% | `grid_stride_loop` 固定核数，减少 launch 开销 |
| Cube/Vector 异构流水 | exp_matmul_sched_004 | 30% | `dl.async_task` Cube/Vector 并行 pipeline |
| 批量矩阵乘 (BMM) | exp_matmul_sched_005 | — | batch 维度 grid 扩展 |
| Grouped GEMM (MoE) | exp_matmul_sched_006 | 25% | 动态 work-stealing 均衡专家负载 |
| Block Swizzling | exp_matmul_sched_007 | 20% | Block 级别 swizzle 提升 L2 局部性 |

### 融合变体

| 融合模式 | 经验 ID | 提升 | 说明 |
|----------|---------|------|------|
| GEMM + Activation (GELU/SiLU) | exp_matmul_fused_activation_001 | 30% | 单 kernel 完成，避免写回再读 |
| GEMM + Bias | exp_matmul_fused_bias_001 | 15% | fixpipe 后处理 |
| GEMM + Scale (量化) | exp_matmul_fused_scale_001 | 20% | 量化 scale 融合 |
| GEMM + RoPE | exp_matmul_fused_rope_001 | 25% | 投影后直接旋转位置编码 |
| GEMM + RMSNorm | exp_matmul_fused_rmsnorm_001 | 30% | 归一化融合 |

### 硬件特定优化

| 场景 | 经验 ID | 提升 | 核心技术 |
|------|---------|------|----------|
| 硬件级对角 PID 映射 | exp_matmul_hw_001 | 30% | `grouped_launch_diagonal` |
| K 维度 padding only | exp_matmul_hw_002 | 15% | `dl.compile_hint("dot_pad_only_k")` |
| HiVM 自动均衡 | exp_matmul_hw_003 | 20% | `enable_hivm_auto_cv_balance` flag |
| Cube/Vector 分离调度 | exp_matmul_hw_004 | 35% | 异构并行，Cube 与 Vector 单元独立调度 |

## 关键技术

### L0 级 Tiling (128×128×128)

Ascend Cube 单元的 L0A/L0B 缓存粒度为 128 字节，最优 Tile 尺寸对齐 L0 访存边界。标准配置：

```
BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 128
```

中间结果 `acc` 保留在 L0C 寄存器中，整个 K 轴迭代完成后一次性写回 L1/GM，避免反复加载。

大矩阵场景（M≥256, N≥256, K≥1024）推荐扩展为 `BLOCK_M=256, BLOCK_N=256, BLOCK_K=256`。

### 对角线 Tiling (Diagonal Block Traversal)

默认行列优先的 PID 映射会导致相邻 core 同时访问同一行/列的 GM，产生 bank conflict。对角线遍历将 PID 映射改为：

```
# 对角线 PID 映射（适合 M,N ≥ 512）
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

实测消除 bank conflict 后提升 25%，多核对角划分（24 核）可达 35%。

### PID 分组与 L2 局部性

通过 `GROUP_SIZE` 控制相邻 PID 的 Core 分配，使同组 core 访问 A 矩阵的同一行块，共享 L2 缓存中的 A 数据：

- `skill_tiling_default_col`：列优先分组，适合 N 维度频繁访问（如 weight 矩阵较宽）
- `skill_tiling_default_row`：行优先分组，适合 M 维度频繁访问（如 batch 较大）
- 推荐 `GROUP_SIZE ∈ {8, 16, 32}`，通过 AutoTune 选取最优值

### NZ Layout + K 循环展开

NZ（列主序分块）格式减少 Cube 单元的数据搬运开销，配合 4× K 循环展开隐藏访存延迟：

```python
# 4x 展开示意
for k in range(0, K, 4 * BLOCK_K):
    a0 = tl.load(a_ptr + k * stride_ak, ...)
    a1 = tl.load(a_ptr + (k+BLOCK_K) * stride_ak, ...)
    a2 = tl.load(a_ptr + (k+2*BLOCK_K) * stride_ak, ...)
    a3 = tl.load(a_ptr + (k+3*BLOCK_K) * stride_ak, ...)
    acc += tl.dot(a0, b0) + tl.dot(a1, b1) + tl.dot(a2, b2) + tl.dot(a3, b3)
```

### 2D Block Pointer (`tl.make_block_ptr`)

使用 `tl.make_block_ptr` 替代手动计算偏移，编译器可生成更优的地址计算指令：

```python
a_block_ptr = tl.make_block_ptr(
    base=a_ptr,
    shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0),  # 行主序
)
```

### Cube/Vector 异构流水 (`dl.async_task`)

将 Cube 计算（矩阵乘）与 Vector 后处理（激活、量化）通过 `dl.async_task` 分配到不同单元并行执行，实现 30% 提升：

```python
with dl.async_task(0):   # Cube 单元：矩阵乘
    acc = tl.dot(a_tile, b_tile)
with dl.async_task(1):   # Vector 单元：激活函数
    out = tl.sigmoid(prev_acc)
```

### 910b2 Dot API 数据格式规范 (skill_dot_input)

在 910b2 上，`tl.dot` 对输入 tensor 的内存布局有严格要求：

- A 矩阵：`[BLOCK_M, BLOCK_K]`，行主序
- B 矩阵：`[BLOCK_K, BLOCK_N]`，行主序（若 trans_b，需提前转置）
- 输出累加器 dtype 推荐 `float32`，最终转换为目标 dtype

违反格式规范会导致静默数值错误，需在实现前通过 `skill_dot_input` 规范检查。

### AutoTiling 搜索空间

```python
# autotiling_enum 推荐搜索空间
BLOCK_M_choices = [128, 256]
BLOCK_N_choices = [128, 256]
BLOCK_K = 256          # 固定，K 维度对齐 Cube 粒度
GROUP_SIZE_choices = [8, 16, 32]
```

通过 `triton.autotune` 或外部脚本枚举所有组合，选取最低延迟配置。

## 常见陷阱

- **Tile 尺寸不对齐**：BLOCK_M / BLOCK_N 非 128 的倍数会导致 Cube 利用率骤降，务必确保 `BLOCK_M % 128 == 0`。
- **K 维度 padding 错误**：非 BLOCK_K 倍数的 K 需显式 padding，推荐使用 `dl.compile_hint("dot_pad_only_k")` 仅对 K 维度 pad，避免 M/N 浪费。
- **双转置未走专项路径**：`trans_a=True, trans_b=True` 不等于两次单转置叠加，需使用 exp_matmul_005 专项配置（M=256, K=256, N=128），否则性能损失显著。
- **NZ layout 混用**：NZ 格式与 ND 格式的指针步长计算不同，混用会导致数据错位，应在进入 kernel 前统一格式。
- **GROUP_SIZE 过大**：GROUP_SIZE > 核心数会导致 L2 局部性反而变差，建议 GROUP_SIZE ≤ 32。
- **忽略 910b2 Dot API 格式**：A/B 矩阵内存布局不符合规范会产生静默数值错误，必须用 `skill_dot_input` 规范检查。
- **Persistent Kernel 核数设置**：`grid_stride_loop` 的固定核数应与硬件实际可用核数匹配（910b2 = 24 核），过少浪费算力，过多引发调度竞争。
- **融合激活未使用 fixpipe**：手写 Vector 激活未走 Cube fixpipe 路径时，Cube/Vector 无法异构并行，丧失融合收益。

## 代码模板

以下为带 AutoTune 的标准 GEMM kernel 模板（ND layout，trans_b=True）：

```python
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn, 'BLOCK_K': 256, 'GROUP_SIZE': gs},
                      num_stages=2, num_warps=4)
        for bm in [128, 256]
        for bn in [128, 256]
        for gs in [8, 16, 32]
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # 列优先 PID 分组（L2 局部性）
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2D Block Pointer
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )

    # L0C 累加器（寄存器复用）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # 写回
    c = acc.to(tl.float16)
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_, N = b.shape
    assert K == K_
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

## 相关页面

- techniques/core_mapping — Core 映射策略（列优先、行优先、对角线）详解
- techniques/tiling — Tiling 策略通用框架与 AutoTiling 方法
- techniques/memory_access — L0/L1/L2 缓存层次与 Block Pointer 使用规范
- techniques/pipeline — Cube/Vector 异构流水线与 `dl.async_task` 使用
- techniques/fusion — 算子融合模式（GEMM+Activation、GEMM+Norm 等）
- [hardware/ascend_910b](ascend-910b.md) — 910b2/910b3 Cube 单元规格、L0 容量与 NZ layout 说明
- [ops/attention](op-class-attention.md) — Flash Attention（GEMM 的重要应用场景）


## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `001-matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/001-matmul_kernel.py`）
- `031-matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/031-matmul_kernel.py`）
- `040-matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/040-matmul_kernel.py`）
- `042-matmul_qn_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/042-matmul_qn_kernel.py`）
- `044-fused_matmul_qn_and_rotary_pos_emb_qr_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/044-fused_matmul_qn_and_rotary_pos_emb_qr_kernel.py`）
- `047-matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/047-matmul_kernel.py`）
- `049-matmul_qn_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/049-matmul_qn_kernel.py`）
- `050-fused_matmul_qn_and_rotary_pos_emb_qk_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/050-fused_matmul_qn_and_rotary_pos_emb_qk_kernel.py`）
- `056-_tensor_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/056-_tensor_matmul_kernel.py`）
- `057-_single_rope_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/057-_single_rope_matmul_kernel.py`）
- `058-_rope_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/058-_rope_matmul_kernel.py`）
- `066-fused_gemm_scale_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/066-fused_gemm_scale_kernel.py`）
- `067-k_grouped_gemm_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/067-k_grouped_gemm_kernel.py`）
- `070-m_grouped_gemm_bKmajor_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/070-m_grouped_gemm_bKmajor_kernel.py`）
- `071-m_grouped_gemm_bNmajor_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/071-m_grouped_gemm_bNmajor_kernel.py`）
- `072-matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/072-matmul_kernel.py`）
- `073-matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/073-matmul_kernel.py`）
- `074-matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/074-matmul_kernel.py`）
- `077-_mhc_pre_gemm_sqrsum_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/077-_mhc_pre_gemm_sqrsum_kernel.py`）
- `078-partial_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/078-partial_matmul_kernel.py`）
- `080-fused_sigmoid_single_norm_rope_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/080-fused_sigmoid_single_norm_rope_matmul_kernel.py`）
- `081-fused_single_norm_rope_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/081-fused_single_norm_rope_matmul_kernel.py`）
- `082-partial_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/082-partial_matmul_kernel.py`）
- `084-fused_matmul_norm_sigmoid_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/084-fused_matmul_norm_sigmoid_kernel.py`）
- `085-fused_matmul_norm_rotary_emb_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/085-fused_matmul_norm_rotary_emb_kernel.py`）
- `089-_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/089-_matmul_kernel.py`）
- `119-addmm_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/119-addmm_kernel.py`）
- `125-bmm_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/125-bmm_kernel.py`）
- `140-outer_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/140-outer_kernel.py`）
- `141-mm_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/141-mm_kernel.py`）
- `147-matmul_bias_persistent_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/147-matmul_bias_persistent_kernel.py`）
- `148-linear_persistent_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/148-linear_persistent_kernel.py`）
- `195-cube1.py` — extracted kernel（source: `raw_experiences/extracted_kernels/195-cube1.py`）
- `198-cube1.py` — extracted kernel（source: `raw_experiences/extracted_kernels/198-cube1.py`）
- `240-_m_grouped_matmul_bKmajor_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/240-_m_grouped_matmul_bKmajor_kernel.py`）
- `241-_m_grouped_matmul_bNmajor_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/241-_m_grouped_matmul_bNmajor_kernel.py`）
- `242-_k_grouped_matmul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/242-_k_grouped_matmul_kernel.py`）

<!-- okf:related:start -->

# 相关

- 实现示例: [FlashAttentionScoreGrad training-backward on Ascend NPU](../ops/flash-attention-score-grad.md) — 主题相关：共享 matmul
- 实现示例: [FlashAttentionScore training-forward on Ascend NPU](../ops/flash-attention-score.md) — 主题相关：共享 matmul
- 实现示例: [FusedFloydAttentionGrad (Floyd attention backward) on Ascend NPU](../ops/fused-floyd-attention-grad.md) — 主题相关：共享 matmul
- 实现示例: [FusedFloydAttention (Floyd multi-dim attention with dual KV groups) on Ascend NPU](../ops/fused-floyd-attention.md) — 主题相关：共享 matmul
- 实现示例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../ops/mla-preprocess.md) — 主题相关：共享 matmul
- 实现示例: [MlaPrologV3 (MLA pre-process) on Ascend NPU](../ops/mla-prolog-v3.md) — 主题相关：共享 matmul
- 实现示例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../ops/mla-prolog.md) — 主题相关：共享 matmul
- 实现示例: [RainFusionAttention (CATLASS block-sparse inference attention) on Ascend NPU](../ops/rain-fusion-attention.md) — 主题相关：共享 matmul
- 实现示例: [RecurrentGatedDeltaRule (linear attention / SSM-style gated delta rule) on Ascend NPU](../ops/recurrent-gated-delta-rule.md) — 主题相关：共享 matmul
- 讲解接口: [GEMM 操作数保持 low_dtype，fp32 累加](../runbooks/algorithm-numerics/gemm-low-dtype-fp32-accum.md) — 主题相关：共享 gemm
- 讲解接口: [matmul core mapping 策略选择（列/行/对角）](../runbooks/compute-pipeline/core-mapping-strategy-selection.md) — 主题相关：共享 matmul
- 讲解接口: [大矩阵 matmul 用对角 core mapping 消除 L2 bank conflict](../runbooks/compute-pipeline/diagonal-core-mapping-l2-bank-conflict.md) — 主题相关：共享 matmul
- 讲解接口: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 gemm
- 讲解接口: [matmul 必须用 tl.make_block_ptr，禁止退化为独立标量 tl.load](../runbooks/memory-sram/block-ptr-vs-scalar-load.md) — 主题相关：共享 matmul
- 讲解接口: [Cube-Vector 场景行宽必须 512B 对齐](../runbooks/memory-sram/row-width-512b-align.md) — 主题相关：共享 matmul
- 讲解接口: [tl.dot vs torch.matmul 精度差异诊断](../runbooks/meta-skills/dot-vs-matmul-precision-diagnosis.md) — 同属优化知识，主题相关
- 讲解接口: [量化 matmul autotune 配置策略诊断--按 shape 范围特化](../runbooks/meta-skills/quant-matmul-autotune-strategy.md) — 同属优化知识，主题相关
- 配套样例: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../runbooks/compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 配套样例: [Matmul Compute 优化聚合](../runbooks/compute-pipeline/traj-matmul-compute.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 配套样例: [Matmul Scheduling 优化聚合](../runbooks/compute-pipeline/traj-matmul-scheduling.md) — 主题相关：共享 910b2, 910b3, gemm, matmul
- 配套样例: [Matmul Split 切分优化：512B 行宽对齐与转置组合分块策略](../runbooks/compute-pipeline/traj-matmul-split.md) — 主题相关：共享 910b2, 910b3, matmul, tiling
- 配套样例: [Matmul 硬件适配优化聚合](../runbooks/portability/traj-matmul-hardware_adaptation.md) — 主题相关：共享 910b2, 910b3, matmul
- 配套样例: [Gemm Basic Skeleton](skeleton-gemm_basic.md) — 同属优化知识，主题相关
- 相关主题: [Fused Attention / Flash Attention 高性能算子](op-class-attention.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [卷积算子优化（Causal Conv1D）](op-class-convolution.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Elementwise 算子优化](op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [融合算子综合页](op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, gemm, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归约算子优化](op-class-reduction.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Softmax / LogSoftmax 算子优化](op-class-softmax.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [SSM / 状态空间模型算子优化](op-class-ssm.md) — 主题相关：共享 910b2, 910b3, operator_class

<!-- okf:related:end -->
