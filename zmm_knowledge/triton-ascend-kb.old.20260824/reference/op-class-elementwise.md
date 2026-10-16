---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: Elementwise 算子优化
description: Elementwise 算子优化
tags: [elementwise, vectorization, fusion, exp, relu, gelu, mul, add, 910b2, 910b3, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/elementwise.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/elementwise.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/elementwise.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# Elementwise 算子优化

## 概述

Elementwise（逐元素）算子对输入张量的每个元素独立执行相同的计算，输出张量与输入张量形状完全相同。典型算子包括激活函数（exp、relu、gelu、silu）、逐元素算术（mul、add、sub、div）和逐元素比较。

由于每个元素的计算完全独立，elementwise 算子天然适合向量化并行，是 Triton-Ascend 内核优化中收益最高的一类场景。优化重点在于：最大化内存带宽利用率（向量化加载/存储）、通过算子融合消除中间张量读写、以及利用 ILP（指令级并行）提升计算吞吐。

## 检测标准

满足以下全部条件时，将算子归类为 elementwise：

- PyTorch 原语为逐元素操作：`torch.exp`、`torch.relu`、`F.gelu`、`F.silu`、`*`（张量乘）、`+`（张量加）等，**不含** `dim=` 归约参数
- 输入张量与输出张量形状完全相同（或满足广播规则后等价）
- 无跨元素的数据依赖（即元素 i 的输出不依赖元素 j 的输入，i≠j）
- 操作可完全向量化，无分支或稀疏访问模式

## 适用条件

| 条件 | 说明 |
|------|------|
| 数据类型 | fp16、bf16、fp32；int8（量化场景）；int64 需特殊处理（见陷阱） |
| 张量形状 | 任意维度，元素总数 N 通常 ≥ 1024 时向量化收益显著 |
| 内存布局 | 连续内存布局收益最大；非连续布局需先调用 `.contiguous()` |
| 算子连续性 | 多个连续 elementwise 算子优先考虑融合为单一内核 |
| 与 matmul 关系 | 紧跟 matmul 之后的激活函数优先内联进 matmul 内核 |

## 经验路由

| 场景 | 经验ID | 提升% | 关键技术 | 来源文件 |
|------|--------|-------|----------|----------|
| 大张量连续加载/存储 | exp_elem_vec_001 | 30% | BLOCK_SIZE=1024/2048 向量化连续 load/store | vectorization.md |
| 非连续内存输入 | exp_elem_vec_002 | 15% | 内核前调用 `.contiguous()` 确保连续内存 | vectorization.md |
| 不同规模张量自适应 | exp_elem_vec_003 | 20% | BLOCK_SIZE autotuning {256,512,1024,2048,4096} | vectorization.md |
| 边界 padding 开销 | exp_elem_vec_004 | 15% | `tl.load()` 设 `other=0`，`boundary_check` 减少同步 | vectorization.md |
| 提升指令级并行 | exp_elem_vec_005 | 20% | For-loop tiling：单次 load+compute+store 拆为多小 tile 循环 | vectorization.md |
| int64 比较算子 | exp_elem_vec_006 | 25% | int64→fp32 类型转换（int64 CMP 退化为标量） | vectorization.md |
| 小算子 launch overhead | exp_elem_vec_007 | 20% | Persistent kernel：固定 grid，grid-stride 循环 | vectorization.md |
| 激活函数紧跟 matmul | exp_elem_fusion_001 | 35% | 激活内联进 matmul 内核，消除独立 elementwise kernel | techniques/operator_fusion，经验：exp_gelu_fwd_001 |
| 多个连续逐点算子 | exp_elem_fusion_002 | 25% | 2-3 个连续 pointwise 算子合并为单内核 | techniques/operator_fusion |
| 中间张量读写瓶颈 | exp_elem_fusion_003 | 30% | 消除中间张量，避免写回 HBM 后再读取 | techniques/operator_fusion |
| 激活+量化组合 | exp_elem_fusion_004 | 30% | 激活与量化在单内核内融合完成 | techniques/operator_fusion，经验：exp_swiglu_quant_001 |
| RoPE 旋转位置编码 | exp_elem_fusion_005 | 25% | elementwise 复数旋转，单内核实现 RoPE | techniques/operator_fusion，经验：experiences/rope/ |

## 关键技术

### 向量化加载/存储（BLOCK_SIZE 选择）

Triton-Ascend 的向量化效率与 BLOCK_SIZE 强相关。BLOCK_SIZE 应选择 2 的幂次，并与硬件向量宽度对齐。

推荐使用 autotuning 在 `{256, 512, 1024, 2048, 4096}` 中搜索最优值：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def elementwise_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.exp(x)  # 替换为目标操作
    tl.store(y_ptr + offsets, y, mask=mask)
```

### care_padding=False 优化

当 padding 不影响计算结果时（如最后一个 tile 的越界元素不参与后续计算），可设置 `other=0` 并省略 boundary check，避免 MTE2/Vector 同步屏障：

```python
# 有 padding 开销（默认）
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# 无 padding 优化（适用于结果不依赖边界值的场景）
x = tl.load(x_ptr + offsets, mask=mask)
```

### For-loop Tiling 提升 ILP

将单次大 tile 的 load→compute→store 拆分为多个小 tile 的循环，使硬件流水线同时处理多个 tile 的不同阶段：

```python
@triton.jit
def elementwise_ilp_kernel(x_ptr, y_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr,
                            NUM_TILES: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE * NUM_TILES
    for i in tl.static_range(NUM_TILES):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.exp(x)
        tl.store(y_ptr + offsets, y, mask=mask)
```

### int64 类型转换

Ascend 910b2/910b3 的 Vector 单元对 int64 比较操作没有原生 SIMD 支持，会退化为逐元素标量计算。将 int64 转换为 fp32 后再做比较可恢复向量化：

```python
# 低效：int64 CMP 退化为标量
result = tl.where(x_int64 > threshold_int64, a, b)

# 高效：转换为 fp32 后向量化比较
x_fp32 = x_int64.to(tl.float32)
threshold_fp32 = tl.full([BLOCK_SIZE], threshold_int64, dtype=tl.float32)
result = tl.where(x_fp32 > threshold_fp32, a, b)
```

### Persistent Kernel（持久化内核）

当张量较小或 launch overhead 不可忽略时，使用持久化内核：固定 grid 为 AI Core 数量，内部使用 grid-stride 循环处理所有数据：

```python
@triton.jit
def persistent_elementwise_kernel(x_ptr, y_ptr, n_elements,
                                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    # grid-stride 循环
    for start in range(pid * BLOCK_SIZE, n_elements, num_programs * BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.exp(x)
        tl.store(y_ptr + offsets, y, mask=mask)

# 启动时固定 grid 为 AI Core 数
grid = lambda meta: (NUM_AI_CORES,)
```

### 算子融合策略

**激活内联进 matmul**（exp_elem_fusion_001，35%提升）：

matmul 输出在寄存器中直接应用激活函数，无需写回 HBM 再读取：

```python
@triton.jit
def matmul_relu_fused_kernel(...):
    # ... matmul 计算 ...
    acc = tl.dot(a_tile, b_tile)      # 结果在寄存器中
    acc = tl.maximum(acc, 0.0)         # relu 内联，无 HBM 读写
    tl.store(c_ptr + offsets, acc)
```

**多 elementwise 融合**（exp_elem_fusion_002，25%提升）：

```python
@triton.jit
def fused_mul_add_relu_kernel(x_ptr, w_ptr, b_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # 三个 pointwise 操作在单内核内完成，无中间 HBM 读写
    y = tl.maximum(x * w + b, 0.0)
    tl.store(y_ptr + offsets, y, mask=mask)
```

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| int64 向量化失效 | Ascend 无原生 int64 SIMD 指令，CMP 退化为标量 | 转换为 fp32 再操作（exp_elem_vec_006） |
| 非连续内存布局 | 步幅不规则导致 gather-like 访问，带宽损失严重 | 内核前调用 `.contiguous()`（exp_elem_vec_002） |
| BLOCK_SIZE 过小 | 无法填满向量单元，利用率低 | autotuning 搜索，最小不低于 256 |
| BLOCK_SIZE 过大 | 超出 UB 容量导致 spill 或编译失败 | 910b2 UB ~192KB，按 dtype 计算最大 BLOCK_SIZE |
| mask 同步开销 | 每次 tl.load 都做 boundary check 引入同步屏障 | 当 N 是 BLOCK_SIZE 整数倍时去掉 mask |
| 独立激活内核 | matmul 后单独启动激活内核，HBM 往返开销高 | 将激活内联进 matmul 内核（exp_elem_fusion_001） |
| 忽略 broadcast 内存访问 | 广播操作可能导致重复加载相同数据 | 使用寄存器缓存广播值，避免重复 load |

## 代码模板

### 通用 Elementwise 模板

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': bs}) for bs in [256, 512, 1024, 2048, 4096]
    ],
    key=['n_elements'],
)
@triton.jit
def elementwise_template_kernel(
    x_ptr,          # 输入指针
    y_ptr,          # 输出指针
    n_elements,     # 元素总数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # ===== 在此替换为目标操作 =====
    y = tl.exp(x)
    # ================================

    tl.store(y_ptr + offsets, y, mask=mask)


def elementwise_op(x: torch.Tensor) -> torch.Tensor:
    # 确保连续内存布局
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    elementwise_template_kernel[grid](x, y, n_elements)
    return y
```

### 多算子融合模板

```python
@triton.jit
def fused_elementwise_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # 多个 pointwise 操作融合，无中间 HBM 读写
    tmp = x * w + b                    # mul + add
    y = tl.sigmoid(tmp) * tmp          # SiLU 激活

    tl.store(y_ptr + offsets, y, mask=mask)
```

## 相关页面

- [normalization.md](op-class-normalization.md) — 归一化算子（LayerNorm、RMSNorm），elementwise 操作在其计算流程中大量出现
- [reduction.md](op-class-reduction.md) — 归约算子，与 elementwise 常组合使用（如 softmax = exp + sum_reduce + div）
- [gather_scatter.md](op-class-gather_scatter.md) — 索引访问算子，elementwise 可与 gather/scatter 融合用于 MoE 路由后的激活计算

## 激活函数反向传播经验

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| SiLU 反向 | exp_silu_bwd_001 | dSiLU = σ(x)·(1 + x·(1-σ(x)))·dy | yaml |
| GELU 反向 | exp_gelu_bwd_001 | dGELU = CDF(x)·dy + x·pdf(x)·dy | yaml |
| GEGLU 反向 | exp_geglu_bwd_001 | GEGLU 门控乘法的反向 | yaml |
| SwiGLU 反向 | exp_swiglu_bwd_001 | SwiGLU 反向，含 SiLU 和 gate 梯度 | yaml |

## 激活函数布局变体经验

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| GEGLU Flat 布局 | exp_geglu_flat_001 | 展平输入布局的 GEGLU 前向 | yaml |
| SwiGLU Flat 布局 | exp_swiglu_flat_001 | 展平输入布局的 SwiGLU 前向 | yaml |
| SwiGLU bwd Flat | exp_swiglu_bwd_flat_001 | 展平布局 SwiGLU 反向 | yaml |
| SwiGLU OAI 格式 | exp_swiglu_oai_001 | OpenAI 格式 SwiGLU（权重分割方式不同） | yaml |
| Sigmoid Grid-Stride | exp_sigmoid_grid_stride_001 | 持久化 sigmoid，grid-stride loop | yaml |

## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `079-sigmoid_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/079-sigmoid_kernel.py`）
- `096-_geglu_forward_kernel_flat.py` — extracted kernel（source: `raw_experiences/extracted_kernels/096-_geglu_forward_kernel_flat.py`）
- `097-_geglu_backward_kernel_flat.py` — extracted kernel（source: `raw_experiences/extracted_kernels/097-_geglu_backward_kernel_flat.py`）
- `105-_swiglu_forward_kernel_flat.py` — extracted kernel（source: `raw_experiences/extracted_kernels/105-_swiglu_forward_kernel_flat.py`）
- `106-_swiglu_backward_kernel_flat.py` — extracted kernel（source: `raw_experiences/extracted_kernels/106-_swiglu_backward_kernel_flat.py`）
- `121-weight_bias_backward_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/121-weight_bias_backward_kernel.py`）
- `142-masked_fill_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/142-masked_fill_kernel.py`）
- `143-masked_fill_kernel_self.py` — extracted kernel（source: `raw_experiences/extracted_kernels/143-masked_fill_kernel_self.py`）
- `146-_swiglu_quant_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/146-_swiglu_quant_kernel.py`）
- `152-fused_sigmoid_gating_delta_rule_update_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/152-fused_sigmoid_gating_delta_rule_update_kernel.py`）
- `178-swiglu_oai_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/178-swiglu_oai_kernel.py`）
- `179-fused_sigmoid_gating_delta_rule_update_npu_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/179-fused_sigmoid_gating_delta_rule_update_npu_kernel.py`）
- `193-_swiglu_forward_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/193-_swiglu_forward_kernel.py`）
- `194-_swiglu_backward_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/194-_swiglu_backward_kernel.py`）
- `211-_swiglu_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/211-_swiglu_fwd_kernel.py`）
- `212-_swiglu_bwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/212-_swiglu_bwd_kernel.py`）
- `214-_silu_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/214-_silu_fwd_kernel.py`）
- `215-_silu_bwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/215-_silu_bwd_kernel.py`）
- `243-_gelu_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/243-_gelu_fwd_kernel.py`）
- `244-_gelu_bwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/244-_gelu_bwd_kernel.py`）
- `247-_element_mul_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/247-_element_mul_kernel.py`）

<!-- okf:related:start -->

# 相关

- 实现示例: [Elementwise absolute value (Abs)](../ops/elementwise-abs.md) — 主题相关：共享 elementwise
- 实现示例: [Elementwise add with a tail mask](../ops/elementwise-add.md) — 主题相关：共享 add, elementwise
- 实现示例: [LightningIndexer (sparse attention indexer) on Ascend NPU](../ops/lightning-indexer.md) — 主题相关：共享 relu
- 实现示例: [MlaPreprocess (MLA attention preprocessing fusion) on Ascend NPU](../ops/mla-preprocess.md) — 主题相关：共享 fusion
- 实现示例: [MlaProlog (MLA attention preprocessing — prolog V1) on Ascend NPU](../ops/mla-prolog.md) — 主题相关：共享 fusion
- 实现示例: [MoeFinalizeRoutingV2Grad (MoE routing finalize backward) on Ascend NPU](../ops/moe-finalize-routing-v2-grad.md) — 主题相关：共享 elementwise
- 实现示例: [MoeFinalizeRoutingV2 (MoE FFN output merge) on Ascend NPU](../ops/moe-finalize-routing-v2.md) — 主题相关：共享 elementwise
- 讲解接口: [跨 tile 归约用 tl.atomic_add 作为可靠原语](../runbooks/algorithm-numerics/atomic-add-cross-tile-reduction.md) — 同属优化知识，主题相关
- 讲解接口: [exp 稳定化 — softmax/rmsnorm 必须减最大值、fp32 累加](../runbooks/algorithm-numerics/exp-stable-softmax-rmsnorm.md) — 同属优化知识，主题相关
- 讲解接口: [dl.async_task 实现 Cube/Vector 异构并行](../runbooks/compute-pipeline/async-task-cube-vector.md) — 主题相关：共享 fusion
- 讲解接口: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 主题相关：共享 fusion
- 讲解接口: [attention 单 kernel 融合失败诊断--softmax 需 workspace，大 D UB 放不下](../runbooks/meta-skills/attention-fusion-failure-diagnosis.md) — 同属优化知识，主题相关
- 讲解接口: [量化融合算子 int8 输出精度不一致的逐层下钻诊断链](../runbooks/meta-skills/quant-fusion-int8-precision-diagnosis.md) — 同属优化知识，主题相关
- 配套样例: [低精度数值稳定性：exp 最大偏移与 sqrt 非负保护](../runbooks/algorithm-numerics/traj-general_stability-numerical_stability.md) — 主题相关：共享 910b2, 910b3, exp
- 配套样例: [Normalization 内存访问优化：Tiling、Two-Pass 与 Kernel Fusion](../runbooks/algorithm-numerics/traj-normalization-memory_access.md) — 主题相关：共享 910b2, 910b3
- 配套样例: [exp_matmul_vec_parallel_001 — matmul 向量并行 + activation fusion](../runbooks/compute-pipeline/instances/traj-exp-matmul-vec-parallel-001.md) — 主题相关：共享 910b2, 910b3
- 配套样例: [Activation Scheduling 优化聚合](../runbooks/compute-pipeline/traj-activation-scheduling.md) — 主题相关：共享 910b2, 910b3, vectorization
- 配套样例: [Scatter Weight Gradient 向量化计算优化](../runbooks/compute-pipeline/traj-gather_scatter-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 配套样例: [Element-wise 算子避免冗余 Reshape](../runbooks/compute-pipeline/traj-general_perf-redundancy_elim.md) — 主题相关：共享 910b2, 910b3, elementwise
- 配套样例: [Misc 算子计算优化：向量化、微内核与阶段融合](../runbooks/compute-pipeline/traj-misc-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 配套样例: [MoE 稀疏 Weight Gradient 计算优化](../runbooks/compute-pipeline/traj-moe-compute.md) — 主题相关：共享 910b2, 910b3, vectorization
- 配套样例: [Gather/Scatter 内存访问优化聚合](../runbooks/memory-sram/traj-gather_scatter-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 配套样例: [Sum-and-Scale 归约内存访问优化](../runbooks/memory-sram/traj-reduction-memory_access.md) — 主题相关：共享 910b2, 910b3, vectorization
- 相关主题: [Fused Attention / Flash Attention 高性能算子](op-class-attention.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [卷积算子优化（Causal Conv1D）](op-class-convolution.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [融合算子综合页](op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, fusion, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归约算子优化](op-class-reduction.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Softmax / LogSoftmax 算子优化](op-class-softmax.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [SSM / 状态空间模型算子优化](op-class-ssm.md) — 主题相关：共享 910b2, 910b3, operator_class

<!-- okf:related:end -->
