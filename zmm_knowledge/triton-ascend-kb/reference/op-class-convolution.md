---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: 卷积算子优化（Causal Conv1D）
description: 卷积算子优化（Causal Conv1D）
tags: [conv1d, causal_conv, ssm, mamba, depthwise, 910b2, 910b3, convolution, operator_class]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/convolution.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/ops/convolution.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/ops/convolution.md` 迁移而来，原始为算子类别设计页，未做内容改写，仅转 OKF frontmatter。

# 卷积算子优化（Causal Conv1D）

## 概述

当前 Triton-Ascend 中主要关注 **Causal Conv1D**（因果一维深度卷积），主要用于 SSM/Mamba 架构中的序列特征提取层。

Causal Conv1D：`y_t = Σ_{k=0}^{W-1} w_k · x_{t-k}`，仅依赖当前及过去时间步（因果性）。

特点：
- **深度卷积**（depthwise）：每个 channel 独立卷积，无 cross-channel 计算
- **小卷积核**：W 通常为 2/4，核参数少，计算量低，带宽密集型
- **SSM 前缀**：通常位于 SSM block 的 input projection 之后，gate 之前

## 经验路由

| 场景 | 经验ID | 关键技术 | 经验文件 |
|------|--------|----------|----------|
| Causal Conv1D 前向 | exp_causal_conv1d_001 | 1D grid 展平调度；tile 加载时包含左侧 (W-1) 个历史元素 | yaml |
| Causal Conv1D 反向 | exp_causal_conv1d_bwd_001 | 权重梯度 + 输入梯度，反向因果关系处理 | yaml |
| Conv1D + 状态更新 | exp_causal_conv1d_states_001 | 与 SSM 状态同步更新，消除中间写回 | yaml |
| **Prefill: 滑动窗口寄存器复用** | exp_causal_conv1d_prefill_002 | col0-3 寄存器移位，每 token 仅 1 次 load（消除 75% load） | yaml |
| **Prefill: 2D Grid + BTD 布局** | exp_causal_conv1d_prefill_003 | `(batch, D-tiles)` 并行 + BTD 连续访问；遗传搜索 1.60x 加速 | yaml |
| **Prefill: 指针基址预计算** | exp_causal_conv1d_prefill_004 | 循环不变量外提；循环内仅 +i*stride 偏移 | yaml |
| **Prefill: Pad slot 提前退出** | exp_causal_conv1d_prefill_005 | 填充序列 1 load+cmp 即退出；释放 compute unit | yaml |
| **Prefill: Conv+Bias+SiLU 融合** | exp_causal_conv1d_prefill_006 | 三合一融合；bias 寄存器驻留；fp32 SiLU 精度保证 | yaml |
| **Prefill: DIM_IS_MULTIPLE** | exp_causal_conv1d_prefill_007 | 原始出处；BLOCK_D 整除 dim 时消除 mask | yaml |
| 增量推理（decode 单步） | exp_causal_conv1d_update_001 | 单步 decode：加载历史 (W-1) 状态，计算当前输出 | yaml |
| W=4 循环展开 + D阈值控制 | exp_causal_conv1d_update_002 | W_IS_4 编译期分支；D≥1024 时 3.1x 加速；D≤256 时自动回退 | yaml |
| block_ptr vs scalar load | exp_causal_conv1d_update_003 | make_block_ptr 硬件 DMA 原语 vs 标量 load 的根因差异；混合方案 | yaml |
| 重叠 T-tiling 消除冗余加载 | exp_causal_conv1d_update_004 | 相邻 tile W-1 重叠区保证卷积窗口连续；仅首 tile 读状态 | yaml |
| DIM_IS_MULTIPLE mask 消除 | exp_causal_conv1d_update_005 | 编译期 mask elision；从 Prefill 迁移的通用优化 | yaml |
| Prefill→Update 知识迁移 | exp_causal_conv1d_update_006 | Bias 融合、指针预计算、单 token 特化、Token-loop vs W-loop 决策 | yaml |
| BDT 布局 + 3D 任务分解 | exp_causal_conv1d_update_007 | 架构决策：布局选择、tile 大小、Grid 策略、双缓冲状态更新 | yaml |

## 关键技术

### 展平网格调度

将 `(batch, channel, time)` 三维任务展平为 1D grid，每个 program 处理一个 `(batch, channel)` 切片的部分时间步：

```python
@triton.jit
def causal_conv1d_fwd(x_ptr, w_ptr, y_ptr, B, D, T, W, BT: tl.constexpr, BD: tl.constexpr):
    pid = tl.program_id(0)
    # 展平：bid = pid // (T//BT // ceil), did = pid // ...
    # 每个 program 处理 BD 个 channel 的 BT 个时间步
```

### 左边界历史元素

因果性要求当前 tile 左侧需加载 (W-1) 个历史元素（padding 或前一 tile 末尾）：

```python
# 加载时包含历史
hist_cols = t_start - W + 1 + tl.arange(0, W - 1)
hist_mask = hist_cols >= 0
hist = tl.load(x_ptr + hist_cols, mask=hist_mask, other=0.0)
# 与当前 tile 拼接后做卷积
```

### 推理单步（增量 decode）

decode 阶段每次仅处理 1 个 token，维护长度为 (W-1) 的卷积历史状态（驻寄存器）：

参考：exp_causal_conv1d_update_001

### Prefill Kernel 架构决策

Prefill kernel（`causal_conv1d_fwd`）针对 Prefill 阶段（长序列，无历史状态）做了专门设计，与 Update kernel 在布局、Grid、T 维度策略上形成对比：

**BTD 布局**：`[B, T, D]` 而非 Update 的 BDT `[B, D, T]`。D 维度连续（stride=1），对逐 token 的滑动窗口卷积更友好——每个 token 的所有 D 维度元素可一次连续 load，无需跨步访问。

**2D Grid 策略**：`grid = (batch, triton.cdiv(dim, BLOCK_D))`，将 batch 和 D-tile 两个维度并行化。不切分 T 维度——滑动窗口寄存器复用机制使 T 维度只需顺序遍历，无需 tile 化。遗传搜索验证此策略带来 1.60x 平均加速。详见 exp_causal_conv1d_prefill_003。

**无 T-tiling 设计**：与 Update kernel 需要重叠 T-tiling 不同，Prefill kernel 不做 T 维度切分。原因：(1) 滑动窗口寄存器复用使逐 token 处理效率极高；(2) T 维度切分会引入 tile 间状态依赖，破坏寄存器复用优势。

### 滑动窗口寄存器复用（Prefill 核心优化）

将卷积窗口的 (W-1) 个历史元素保持在命名寄存器中，每处理一个 token 仅需加载 1 个新元素（而非 W 个），消除 75% 的 HBM 读：

```python
# 初始化：加载窗口前 (W-1) 个历史 + 当前 token
col0 = tl.load(conv_states_ptr + 0*stride, mask=mask_d)
col1 = tl.load(conv_states_ptr + 1*stride, mask=mask_d)
col2 = tl.load(conv_states_ptr + 2*stride, mask=mask_d)
col3 = tl.load(x_ptrs)                          # 当前 token（仅此为新加载）

for i in range(seq_len):
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3  # fp32 FMA
    tl.store(y_ptrs + i * stride_y, acc, mask=mask_d)
    # 寄存器移位：丢弃最旧元素，所有元素左移一位
    col0 = col1; col1 = col2; col2 = col3
    col3 = tl.load(x_ptrs + (i+1) * stride_x, mask=mask_d)  # 仅 1 次新 load
```

**通用性**：此模式适用于所有小固定窗口的序列处理 kernel（W=2-8），如移动平均、指数平滑、因果卷积、SSM 扫描的前缀处理。核心条件：窗口大小固定且足够小，可在寄存器中完整驻留。

参考：exp_causal_conv1d_prefill_002

### 指针基址预计算（循环不变量外提）

将循环不变地址分量（base + start_offset + d_offs * stride）提到循环外预计算，循环内仅做简单偏移 `+ i * stride`：

```python
# 循环外预计算基址指针（不变量外提）
x_ptrs = x_ptr + start * stride_x_tokens + d_offs * stride_x_dim
y_ptrs = y_ptr + start * stride_y_tokens + d_offs * stride_y_dim

# 循环内仅做简单偏移
for i in range(seq_len):
    col3 = tl.load(x_ptrs + i * stride_x, mask=mask_d)
    tl.store(y_ptrs + i * stride_y, acc, mask=mask_d)
```

消除循环内重复的 `start * stride` 和 `d_offs * stride` 地址运算。在 seq_len=2048 场景下节省约 4096 次乘加地址计算。

此模式是通用 LICM（Loop-Invariant Code Motion）优化，适用于所有含循环体的 kernel。

参考：exp_causal_conv1d_prefill_004

### Conv+Bias+SiLU 三合一融合

将卷积计算、Bias 加法、SiLU 激活融合在单 kernel 的每条 token 循环内。所有中间结果保留在寄存器中（fp32 精度），只产生一次最终 store：

```python
# 循环外：bias 一次性加载，寄存器驻留
if HAS_BIAS:
    bias = tl.load(bias_ptr + d_offs, mask=mask_d)

for i in range(seq_len):
    col3 = tl.load(x_ptrs + i * stride_x, mask=mask_d)
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3  # fp32

    if HAS_BIAS:
        acc = acc + bias                          # fp32 bias add
    if ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))            # fp32 SiLU（保证数值稳定性）

    tl.store(y_ptrs + i * stride_y, acc.to(OUT_DTYPE), mask=mask_d)
    col0 = col1; col1 = col2; col2 = col3
```

**关键设计**：(1) bias 向量在循环外一次性加载到寄存器，循环内复用（节省 seq_len-1 次 load）；(2) SiLU 在 fp32 累加器上计算，避免 fp16 下 exp overflow（`x < -88` 时 `exp(-x) → inf`）；(3) `HAS_BIAS` 和 `ACTIVATION` 为 constexpr 开关，不需要时编译器完全消除对应路径。相比 3 个独立 kernel（conv → add_bias → silu），消除 2 次 HBM 写 + 2 次 HBM 读。

参考：exp_causal_conv1d_prefill_006

### Pad Slot 提前退出

对变长 batch 中的 padding 序列（`cache_idx == pad_slot_id`），在 kernel 入口处做单次检查后立即 return，释放 compute unit：

```python
cache_idx = tl.load(cache_indices_ptr + pid_batch)
if cache_idx == pad_slot_id:   # -1
    return                      # 立即释放 compute unit
```

**收益**：1 次 load + 1 次整数比较的开销（~10 cycles）远小于执行完整卷积循环（~seq_len 次 load/store/FMA）。batch 中有 ≥10% 填充序列时收益显著。在 Ascend 的 grid-stride 调度下，提前退出的 program 对应的 tile 可被重新分配给其他 tasks。

**通用性**：适用于所有使用间接索引映射 batch 的 kernel（KV-cache page table、embedding lookup with dynamic indices、MoE routing），只需单次 load + compare 即可判断是否跳过。

参考：exp_causal_conv1d_prefill_005

### DIM_IS_MULTIPLE 原始出处

当 `dim % BLOCK_D == 0`（如 dim=4096, BLOCK_D=1024），通过编译期常量跳过 D 维度的 mask 检查。此经验是 DIM_IS_MULTIPLE 模式的**原始出处**（Prefill kernel Round 2 遗传搜索），与指针预计算组合贡献 26% 性能提升：

```python
# host 端
DIM_IS_MULTIPLE = (dim % BLOCK_D == 0)  # BLOCK_D=1024, dim=4096 → True

# kernel 端
@triton.jit
def kernel(..., DIM_IS_MULTIPLE: tl.constexpr, BLOCK_D: tl.constexpr):
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    if DIM_IS_MULTIPLE:
        mask_d = None       # 所有 load/store 无需边界检查
    else:
        mask_d = d_offs < dim
    x = tl.load(ptr + d_offs, mask=mask_d)
```

免去 mask 后每条 load/store 省去：(1) `d_offs < dim` 向量比较指令；(2) predication 的 mask 寄存器操作；(3) 越界元素的 `other=0.0` 填充逻辑。Update kernel 版本见 exp_causal_conv1d_update_005。

通用性：适用于所有做维度 tile 化的 kernel（matmul、attention、normalization、elementwise、rope），只要 tile size 能整除维度大小。

参考：exp_causal_conv1d_prefill_007

### Update Kernel 架构决策

Update kernel（`causal_conv1d_update_kernel_bdt_fwd`）针对 Decode 阶段（短序列 + 已有状态）做了专门设计：

**BDT 布局**：`[B, D, T]` 而非 Prefill 的 BTD `[B, T, D]`。D 维度连续（stride=1），对短序列的矩阵式卷积更友好。详见 exp_causal_conv1d_update_007。

**3D 任务分解**：1D grid + strided loop 映射 `batch × D-tiles × T-tiles`。D_CHK_SIZE=16（小 tile 保证访问连续）+ T_CHK_SIZE=256（大 tile 减少调度开销）。

**双缓冲状态更新**：kernel 写入独立 `conv_state_update` buffer，host 端 `copy_` 同步，保证多 program 并发写入无竞争。

### W=4 循环展开与 D 阈值控制

当 W=4 时，可展开卷积循环为 4 路独立计算表达式，消除 transpose 和 extract_slice 开销。**但必须配合 D 维度阈值控制**：

| D 范围 | NUM_D_CHK | 加速比 | 策略 |
|--------|-----------|--------|------|
| D ≥ 2048 | ≥128 | **2.3x - 3.1x** | 启用展开 |
| D ≤ 256 | ≤16 | 0.41x - 0.75x | 回退到 for 循环 |

阈值控制一行改动：`W_IS_4=(width==4 and dim>=1024)`。详见 exp_causal_conv1d_update_002。

**负收益根因**：展开后用 4 次独立 `tl.load` 替代 1 次 `make_block_ptr`，丧失硬件 block transfer 加速（exp_causal_conv1d_update_003）；寄存器压力增大约 3 倍。

**混合方案（进阶）**：保留 `make_block_ptr` 加载权重，仅消除 transpose，同时追求大 D 加速和小 D 不退化。

### 重叠 T-tiling

相邻 T-tile 之间有 (W-1) 个元素的重叠区，保证卷积窗口在 tile 边界连续。仅首个 tile 需要从 conv_state 读取历史 token。详见 exp_causal_conv1d_update_004。

### Prefill ↔ Update 跨 kernel 优化迁移

以下优化可从 Prefill kernel 迁移至 Update kernel：

| 优化 | 优先级 | 收益 | 经验文件 |
|------|--------|------|----------|
| HAS_BIAS 支持 | 高 | 消除额外 bias add kernel launch | exp_causal_conv1d_update_006 |
| DIM_IS_MULTIPLE mask 消除 | 高 | 减少 D 维度所有 load/store 的 mask 运算 | exp_causal_conv1d_update_005 |
| 内层循环预计算指针 | 中 | 减少 extract_slice 地址运算 | exp_causal_conv1d_update_006 |
| 单 token decode 滑动窗口特化 | 低 | seq_len=1 时用寄存器复用替代 buffer 加载 | 同上 |

**不适用迁移**：W 硬编码为 4（Update 需通用 W）；2D Grid（Update 需 T-tiling）。

## Prefill vs Update 架构对比

| 维度 | Prefill Kernel | Update Kernel |
|------|---------------|---------------|
| 数据布局 | BTD `[B, T, D]`（D 连续） | BDT `[B, D, T]`（D 连续，短序列友好） |
| Grid 策略 | 2D `(batch, D-tiles)`，无 T-tiling | 1D + 3D 任务分解（含 T-tiling） |
| T 维度处理 | 寄存器滑动窗口（逐 token，无 tile 边界） | 重叠 T-tiling（W-1 重叠区保证连续） |
| 核心优化 | 寄存器复用（75% load 削减）+ 三合一融合 | W=4 展开 + D 阈值控制 + block_ptr |
| 状态管理 | 读取 conv_states 前 (W-1) 个历史元素 | 双缓冲写入 conv_state_update |
| BLOCK_D | 1024（大 tile，利用 BTD 连续性） | 16（小 tile，BDT 下避免跨步） |
| 适用阶段 | Prefill（长序列，seq_len ≥ 512） | Decode（短序列，seq_len=1 或小值） |

## 适用条件

| 参数 | 推荐范围 |
|------|----------|
| D（channel 数） | ≥64，BD=32/64 时带宽利用率最高 |
| W（卷积核宽） | 2、4（Mamba 常用），过大增加历史加载开销 |
| BT（时间步 tile，Prefill） | ≥16，通常 64/128 |
| BLOCK_D（Prefill kernel） | 1024（默认），BTD 布局下大 tile 利用 D 连续访问 |
| T_CHK_SIZE（Update kernel） | 256（默认），需远大于 seq_len |
| D_CHK_SIZE（Update kernel） | 16（默认），小值保证 BDT 布局下访问连续 |
| W_IS_4 展开阈值 | D ≥ 1024 时启用，D < 1024 时回退 |
| Prefill 寄存器复用条件 | W ≤ 8（寄存器可容纳），seq_len ≥ 64（摊还初始化开销） |

## 相关页面

- [ops/ssm.md](op-class-ssm.md) — Causal Conv1D 常与 SSM 状态更新融合
- techniques/tiling.md — BT/BD tile 大小选择；BDT vs BTD 布局驱动的异构 tile 策略；无 T-tiling 决策
- techniques/ilp.md — 滑动窗口寄存器复用（小固定窗口序列通用模式）；循环展开退化分析
- techniques/parallelism_and_loop.md — 2D Grid + 无 T-tiling 模式；1D Grid + 3D 任务分解
- techniques/memory_access_optimization.md — 寄存器驻留 + 移位代替重复加载；make_block_ptr vs scalar load；重叠 T-tiling
- techniques/compute_and_branch.md — constexpr 分支消除（DIM_IS_MULTIPLE、W_IS_4）；指针基址预计算（LICM）；Pad slot 提前退出
- techniques/operator_fusion.md — Conv+Bias+SiLU 三合一融合；Conv1D+State+SiLU 双缓冲融合
- techniques/persistent_kernels.md — decode 单步的 persistent 推理模式

## 已知实例（P3 挂载）

### Extracted Kernel 反链
- `029-_causal_conv1d_update_kernel_no_cache_len_no_mtp.py` — extracted kernel（source: `raw_experiences/extracted_kernels/029-_causal_conv1d_update_kernel_no_cache_len_no_mtp.py`）
- `030-_causal_conv1d_update_kernel_gpu.py` — extracted kernel（source: `raw_experiences/extracted_kernels/030-_causal_conv1d_update_kernel_gpu.py`）
- `133-causal_conv1d_update_kernel_bdt_fwd.py` — extracted kernel（source: `raw_experiences/extracted_kernels/133-causal_conv1d_update_kernel_bdt_fwd.py`）
- `135-radix_type_convert.py` — extracted kernel（source: `raw_experiences/extracted_kernels/135-radix_type_convert.py`）
- `162-_causal_conv1d_update_kernel_npu_tiled.py` — extracted kernel（source: `raw_experiences/extracted_kernels/162-_causal_conv1d_update_kernel_npu_tiled.py`）
- `187-_causal_conv1d_update_kernel_no_cache_len_no_mtp.py` — extracted kernel（source: `raw_experiences/extracted_kernels/187-_causal_conv1d_update_kernel_no_cache_len_no_mtp.py`）
- `188-_causal_conv1d_update_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/188-_causal_conv1d_update_kernel.py`）
- `266-causal_conv1d_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/266-causal_conv1d_fwd_kernel.py`）
- `267-causal_conv1d_bwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/267-causal_conv1d_bwd_kernel.py`）
- `268-causal_conv1d_states_fwd_kernel.py` — extracted kernel（source: `raw_experiences/extracted_kernels/268-causal_conv1d_states_fwd_kernel.py`）

<!-- okf:related:start -->

# 相关

- 讲解接口: [寄存器移位复用滑动窗口，消除重复 HBM 读](../runbooks/memory-sram/register-shift-sliding-window.md) — 主题相关：共享 convolution, ssm
- 配套样例: [Flash Attention Causal Skeleton](skeleton-flash_attention_causal.md) — 同属优化知识，主题相关
- 相关主题: [Fused Attention / Flash Attention 高性能算子](op-class-attention.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Elementwise 算子优化](op-class-elementwise.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Embedding / 索引拷贝算子优化](op-class-embedding.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [融合算子综合页](op-class-fused_ops.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Gather/Scatter 算子优化](op-class-gather_scatter.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Loss 函数算子优化](op-class-loss.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [MatMul / GEMM 高性能算子](op-class-matmul.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归一化算子优化](op-class-normalization.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [归约算子优化](op-class-reduction.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [RoPE（旋转位置编码）算子优化](op-class-rope.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [Softmax / LogSoftmax 算子优化](op-class-softmax.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [solve_tril 算子优化（三角矩阵求逆）](op-class-solve_tril.md) — 主题相关：共享 910b2, 910b3, operator_class
- 相关主题: [SSM / 状态空间模型算子优化](op-class-ssm.md) — 主题相关：共享 910b2, 910b3, mamba, operator_class, ssm

<!-- okf:related:end -->
