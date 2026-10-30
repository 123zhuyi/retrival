---
id: pattern-flash-attention-tiling-key-dispatch
title: FlashAttention multi-template tiling key dispatch on Ascend NPU
type: pattern
tags: [tiling_key, dispatch, flash_attention, attention, template, constexpr]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, l1_buffer, l0c_buffer, cube_vector_1_to_2]
architectures: [Ascend910B1, Ascend910_93, Ascend950PR]
kernel_types: [attention]
operators: [flash_attention_score]
confidence: inferred
reproducibility: snippet
sources: [ops-transformer-flash-attention-score]
symptoms: [performance_bottleneck, compile_error, precision_mismatch]
candidate_techniques: [fp32_accumulation, tail_mask]
indicators: [indicator-18, indicator-19, indicator-20]
related: [technique-flash-attention-tiling, technique-flash-attention-pipeline]
---

# FlashAttention 多模板 tiling key dispatch 模式

FlashAttention 使用 **编译期 tiling key** 在静态编译时展开所有模板组合，运行时由 Host 侧 tiling 计算确定 tiling key 值，kernel 侧通过 `if constexpr` 选择对应模板实例。这种模式将运行时分支转化为编译期特化，消除分支开销，但代价是编译实例数量爆炸。

## 1. 架构差异

| 架构 | tiling key 文件 | bit 宽度 | 字段数 | 模板选择入口 | 编译实例量级 |
|------|----------------|----------|--------|-------------|-------------|
| arch32 (A2) | `arch32/flash_attention_score_template_tiling_key.h` | 64 bit | 17 | `ASCENDC_TPL_ARGS_DECL` + `ASCENDC_TPL_SEL` | ~2300 |
| arch35 (950PR) | `arch35/flash_attention_score_template_tiling_key.h` | 62 bit | 12 | 同上 | ~6000 |

## 2. arch32 tiling key 编码

```
bit:3-0    KernelTypeKey   (4bit): 0=正常, 1=空tensor
bit:7-4    UB0            (4bit): 核内切分轴1 (0=B,1=N2,2=G,3=S1,4=S2,5=D,9=NONE)
bit:11-8   UB1            (4bit): 核内切分轴2 (同上)
bit:15-12  Block          (4bit): 核间切分轴 (同上)
bit:17-16  ImplMode       (2bit): 0=HIGH_PRECISION, 1=HIGH_PERFORMANCE, 2=INVALID_LINE_HP
bit:19-18  DataType       (2bit): 0=FP16, 1=FP32, 2=BF16, 3=FP16_PRECISION
bit:22-20  Layout         (3bit): 0=NONE, 1=BSH, 2=SBH, 3=BNSD, 4=TND
bit:23     Bmm1Format     (1bit): 0=ND, 1=NZ
bit:24     Bmm2Source     (1bit): 0=GM, 1=L1(TSCM)
bit:28-25  Sparse         (4bit): 0=ALL,1=NONE,2=ANY,3=CAUSAL,4=BAND,5=PREFIX,6=BAND_COMPRESS,7-9=其他
bit:30-29  BigDoubleBuffer(2bit): 0=NONE, 1=BIG_BUFFER, 2=BIG_DOUBLE_BUFFER
bit:31     HasDropOut     (1bit): 0/1
bit:32     HasAttenMask   (1bit): 0/1
bit:33     HasPse         (1bit): 0/1
bit:34     EnableL1Reuse  (1bit): 0/1
bit:35     HasRope        (1bit): 0/1
bit:36     MatmulPolicyType(1bit): 0=NORMAL, 1=UNSPLITK
bit:40-37  S1TemplateType (4bit): 0=UNKNOWN, 1-8=ALIGNED_16..ALIGNED_128
bit:44-41  S2TemplateType (4bit): 同上
bit:48-45  DTemplateType  (4bit): 0=UNKNOWN, 5-8=ALIGNED_80..ALIGNED_128
```

### arch32 模板选择与 kernel dispatch

kernel 入口 `flash_attention_score.cpp` 通过 `if constexpr` 对模板参数做分发：

| 条件 (UB0,UB1,Block,Layout) | 模板类 | 说明 |
|------------------------------|--------|------|
| UB0=3, UB1=4, Block=9, Layout=4 | FlashAttentionVarLenScore | TND 模板 |
| UB0=3, UB1=9, Block=9, Layout=4 | FlashAttentionVarLenScoreSameAB | TND SameAB 模板 |
| UB0=3, UB1=9, Block=9, Layout≠4 | FlashAttentionScoreS1s2Bn2gs1SameAB | SameAB 模板 |
| UB0=3, UB1=4, Block=9, Layout≠4 | FlashAttentionScoreS1s2Bn2gs1 | S1S2 模板 |
| UB0=3, UB1=5, Block=9 | FlashAttentionScoreS1Bn2gs1 | S1 模板 |
| UB0=9, UB1=9, Block=0 | FlashAttentionScoreBn2gs1s2B | B 模板 |
| KernelTypeKey=1 | FlashAttentionScoreEmptyTensor | 空 tensor |

### arch32 模板数量分解

**FP16 (DataType=3)**:
| 模板 | 选择数 | 计算方式 |
|------|--------|---------|
| S1S2 ND | 96 | 2(ImplMode)×3(Layout)×2(Sparse)×2(DropOut)×2(AttenMask)×2(Pse)×2(L1Reuse) |
| S1S2 NZ | 96 | 同上 |
| S1 ND+GM | 288 | 2×3×2×2×2×2×3(S1Type) |
| S1 ND+L1 | 288 | 同上 |
| B | 48 | 2×3×2×2×2 |
| SameAB ND | 48 | 同 B |
| SameAB NZ | 48 | 同 B |
| SameAB UnsplitK | 16 | 2×2×2×2 |
| TND | 16 | 同上 |
| TND RoPE | 1 | — |
| TNDSameAB | 8 | 2×2×2 |
| **小计** | **~960** | — |

**BF16 (DataType=2)**: 类似 + scalar const 模板 ≈ 1100
**FP32 (DataType=1)**: S1S2+S1+B+TND ≈ 200
**空 tensor**: 1
**arch32 总计**: **~2300**

## 3. arch35 tiling key 编码

```
bit:1-0    KernelTypeKey   (2bit): 0=正常, 1=空tensor
bit:3-2    ImplMode       (2bit): 同 arch32
bit:7-4    Layout         (4bit): 0=None,1=SBND,2=BNSD,3=TND(?),4=TND
bit:17-8   S1TemplateType (10bit): 0/16/64/128/256
bit:27-18  S2TemplateType (10bit): 0/16/32/64/128/256/512
bit:39-28  DTemplateType  (12bit): 0/16/32/48/64/80/96/128/160/192/256/768
bit:51-40  DvTemplateType (12bit): 同 D 或显式指定
bit:55-52  PseMode        (4bit): 0=OUTER_MUL_ADD,1=OUTER_ADD_MUL,2=INNER_MUL_ADD,3=INNER_MUL_ADD_SQRT,4=INVALID,9=NONE
bit:56     HasAtten       (1bit): 0/1
bit:57     HasDrop        (1bit): 0/1
bit:58     HasRope        (1bit): 0/1
bit:60-59  OutDtype       (2bit): 0/1/2
bit:61     Regbase        (1bit): 0/1
```

### arch35 特点

- **大 bit 宽度的 S1/S2/D/Dv TemplateType**：使用实际维度值（16/32/64/128/256 等）而非序号，精确匹配编译期常量。
- **DvTemplateType**：与 DTemplateType 独立，支持 Q-K D 与 V Dv 不同（GQA/MQA 场景）。
- **PseMode**：5 种模式比 arch32 的 HasPse 更细粒度。
- **OutDtype**：支持 FP8 输出场景。

### arch35 模板数量分解

| dtype | 模板选择 | 说明 |
|------|---------|------|
| BF16/FP16 | ~5000 | 多个 S1/S2/D/Dv/PseMode 组合 × Layout × HasAtten/Drop/Rope |
| FP32 | ~900 | D=768/192/64-256 等 |
| FP8 | ~50 | D=64/128, S1=128, S2=512, PseMode=NONE |
| 空 tensor | 1 | — |

## 4. 通用 dispatch 模式总结

1. **Host 侧 tiling 计算**：`TilingFlashAttentionScore` → 架构检测 → `TilingRegistryArch::DoTilingImpl` → 设置 tiling key + tiling data。
2. **编译期模板展开**：`ASCENDC_TPL_ARGS_DECL` 声明所有 bit 字段，`ASCENDC_TPL_SEL` + `ASCENDC_TPL_ARGS_SEL` 列举每个要编译的具体组合。
3. **Kernel 侧 constexpr 分发**：`if constexpr` 对模板参数做零开销分支，每个编译实例只有一条路径。
4. **宏简化调用**：FA 使用 `INVOKE_FA_GENERAL_OP_IMPL` / `_SAMEAB` / `_BMM1NZ` / `_WITH_MMPOLICY` 等宏封装 tiling data 拷贝 + Matmul 注册 + op.Init + op.Process；SFA 使用单一的 `SFA_OP_IMPL` 宏。

## 4.1 SparseFlashAttention dispatch 入口

SFA 的 kernel 入口通过 `if constexpr` 对 dtype 做分发：

```cpp
// kernel.cpp:32-53
// FP16
using SFAType = SFAType<half, half, half, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE>;
// BF16
using SFAType = SFAType<bfloat16, bfloat16_t, bfloat16_t, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE>;
```

然后通过 `SFA_OP_IMPL` 宏实例化 `SparseFlashAttentionMla` 模板类。与 FA 的 `INVOKE_FA_GENERAL_OP_IMPL` 宏体系不同，SFA 的模板参数更少，dispatch 路径更简单。

## 5. SparseFlashAttention 模板参数

SFA 的模板参数声明（template_tiling_key.h:31-37）：

```cpp
ASCENDC_TPL_ARGS_DECL(SparseFlashAttention,
ASCENDC_TPL_BOOL_DECL(FLASH_DECODE, 0, 1),
ASCENDC_TPL_UINT_DECL(LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND, SFA_LAYOUT_TND),
ASCENDC_TPL_UINT_DECL(KV_LAYOUT_T, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, SFA_LAYOUT_BSND, SFA_LAYOUT_TND,
                      SFA_LAYOUT_PA_BSND),
ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, C_TEMPLATE, V_TEMPLATE),
);
```

### 5.1 bit 字段定义

| bit 字段 | 宽度 | 取值 |
|----------|------|------|
| FLASH_DECODE | 1bit | 0/1 |
| LAYOUT_T | 4bit | BSND(0)/TND(1) |
| KV_LAYOUT_T | 4bit | BSND(0)/TND(1)/PA_BSND(2) |
| TEMPLATE_MODE | 4bit | C_TEMPLATE(0)/V_TEMPLATE(1) |

SFA 只有 **4 个模板参数**，远少于 FA 的 17 个（arch32）或 12 个（arch35）。这是因为 SFA 面向固定 headDim=512+64=576 的 MLA 场景，不需要 S1/S2/D/Dv 的 TemplateType。`C_TEMPLATE`（Cube 主）和 `V_TEMPLATE`（Vector 主）的选择替代了 FA 中复杂的多模板（S1S2 / S1 / B / SameAB / TND / TNDSameAB）选择逻辑。

### 5.2 编译实例组合

SFA 的编译实例通过 `ASCENDC_TPL_ARGS_SEL` 列举（template_tiling_key.h:41-69）：

| layout | TEMPLATE_MODE | KV_LAYOUT |
|--------|---------------|-----------|
| BSND | C_TEMPLATE | BSND, PA_BSND |
| TND | C_TEMPLATE | TND, PA_BSND |
| BSND | V_TEMPLATE | BSND, PA_BSND |
| TND | V_TEMPLATE | TND, PA_BSND |

共 **约 8 个编译实例**（× 2 种 dtype = 16），远少于 FA 的 ~2300 个（arch32）或 ~6000 个（arch35）。精简的原因是：
- **固定 headDim**：无需 D/Dv 的 TemplateType 展开
- **无 S1/S2 模板类型**：S2 由 sparse_indices 动态驱动，S1 固定 mBaseSize=128
- **无 mask/pse/dropout 等特征 bit**：SFA 专注 MLA 稀疏注意力场景

### 5.3 与 FA 的对比

| 维度 | FA (arch32) | SFA |
|------|-------------|-----|
| 模板参数字段数 | 17 | 4 |
| 编译实例量级 | ~2300 | ~8（× 2 dtype = 16） |
| 模板选择策略 | 6 类模板（S1S2/S1/B/SameAB/TND/TNDSameAB） | 2 类模板（C_TEMPLATE/V_TEMPLATE） |
| headDim | 可变（1~768），S1/S2/D/Dv 4 个 TemplateType | 固定 512+64=576，无 TemplateType |
| 特征分支 | HasDropOut / HasAttenMask / HasPse / HasRope / Sparse 等 | 无（由 sparse_indices 动态控制） |
| Dispatch 宏 | `INVOKE_FA_GENERAL_OP_IMPL` / `_SAMEAB` / `_BMM1NZ` 等 | `SFA_OP_IMPL` 单一宏 |

详见 [[technique-sparse-attention-tiling]] 了解 SFA 的 tiling 策略细节。

## 6. 对 Triton-Ascend 的映射

在 Triton-Ascend 中，没有编译期 tiling key 机制，对应的是 `@triton.jit` kernel 的 **constexpr meta 参数**：

```python
@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    BLOCK_M: tl.constexpr,  # ≈ S1TemplateType
    BLOCK_N: tl.constexpr,  # ≈ S2TemplateType
    HEAD_DIM: tl.constexpr, # ≈ DTemplateType
    IS_CAUSAL: tl.constexpr,# ≈ Sparse == CAUSAL
    HAS_MASK: tl.constexpr, # ≈ HasAttenMask
    HAS_PSE: tl.constexpr,  # ≈ HasPse / PseMode
    HAS_DROPOUT: tl.constexpr, # ≈ HasDropOut
    SCALE: tl.constexpr,
):
```

Triton 通过 autotune/config 选择最优 meta 组合，替代 CANN 的 tiling key + 多模板编译模式。

## 8. Agent actions

1. 确定目标架构后查阅对应 tiling key 编码表。
2. 从输入 shape/dtype/layout/mask/pse/dropout 推导 tiling key 各 bit 值。
3. 通过 tiling key 确定激活的 kernel 模板类和 tiling data 结构。
4. Triton 实现时用 `tl.constexpr` meta 参数替代 tiling key，用 autotune 替代编译期展开。
5. arch35 的 DvTemplateType 支持 D≠Dv，Triton 中需两个 constexpr 参数。

来源见 [[ops-transformer-flash-attention-score]]。
