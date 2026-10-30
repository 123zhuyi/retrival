# FlashAttention Template Dispatch Table

Derived from static analysis of CANN ops-transformer FlashAttentionScore (commit d28d8f99).
`confidence: inferred`, `reproducibility: snippet`.

## 1. arch32 (A2) Tiling Key Bit 编码

| Bit 位置 | 宽度 | 字段名 | 值域 | 说明 |
|----------|------|--------|------|------|
| 3-0 | 4 | KernelTypeKey | 0, 1 | 0=正常, 1=空tensor |
| 7-4 | 4 | UB0 | 0-5, 9 | 核内切分轴1: 0=B,1=N2,2=G,3=S1,4=S2,5=D,9=NONE |
| 11-8 | 4 | UB1 | 0-5, 9 | 核内切分轴2 |
| 15-12 | 4 | Block | 0-5, 9 | 核间切分轴 |
| 17-16 | 2 | ImplMode | 0-2 | 0=HIGH_PREC, 1=HIGH_PERF, 2=INVALID_LINE_HP |
| 19-18 | 2 | DataType | 0-3 | 0=FP16, 1=FP32, 2=BF16, 3=FP16_PRECISION |
| 22-20 | 3 | Layout | 0-4 | 0=NONE, 1=BSH, 2=SBH, 3=BNSD, 4=TND |
| 23 | 1 | Bmm1Format | 0, 1 | 0=ND, 1=NZ |
| 24 | 1 | Bmm2Source | 0, 1 | 0=GM, 1=L1 |
| 28-25 | 4 | Sparse | 0-9 | 见 Sparse 模式编码表 |
| 30-29 | 2 | BigDoubleBuffer | 0-2 | 0=NONE, 1=BIG, 2=BIG_DOUBLE |
| 31 | 1 | HasDropOut | 0, 1 | — |
| 32 | 1 | HasAttenMask | 0, 1 | — |
| 33 | 1 | HasPse | 0, 1 | — |
| 34 | 1 | EnableL1Reuse | 0, 1 | — |
| 35 | 1 | HasRope | 0, 1 | — |
| 36 | 1 | MatmulPolicyType | 0, 1 | 0=NORMAL, 1=UNSPLITK |
| 40-37 | 4 | S1TemplateType | 0-8 | 0=UNKNOWN, 1-8=ALIGNED_16..128 |
| 44-41 | 4 | S2TemplateType | 0-8 | 同上 |
| 48-45 | 4 | DTemplateType | 0,5-8 | 0=UNKNOWN, 5-8=ALIGNED_80..128 |

## 2. arch32 模板 Dispatch 条件

| 模板 | tiling key 条件 (if constexpr) | Kernel 类 | tiling 数据结构 |
|------|-------------------------------|-----------|----------------|
| 空 tensor | KernelTypeKey==1 | FlashAttentionScoreEmptyTensor | FlashAttentionScoreTilingData |
| TND SameAB | UB0=3,UB1=9,Block=9,Layout=4 | FlashAttentionVarLenScoreSameAB | FlashAttentionScoreGeneralTilingData |
| TND | UB0=3,UB1=4,Block=9,Layout=4 | FlashAttentionVarLenScore | FlashAttentionScoreGeneralTilingData |
| SameAB | UB0=3,UB1=9,Block=9,Layout≠4 | FlashAttentionScoreS1s2Bn2gs1SameAB | FlashAttentionScoreGeneralTilingData |
| S1S2 | UB0=3,UB1=4,Block=9,Layout≠4 | FlashAttentionScoreS1s2Bn2gs1 | FlashAttentionScoreGeneralTilingData |
| S1 | UB0=3,UB1=5,Block=9 | FlashAttentionScoreS1Bn2gs1 | FlashAttentionScoreGeneralTilingData |
| B | UB0=9,UB1=9,Block=0 | FlashAttentionScoreBn2gs1s2B | FlashAttentionScoreGeneralTilingData |

## 3. arch32 模板数量统计

| dtype | S1S2 | S1 | B | SameAB | TND | TNDSameAB | 小计 |
|-------|------|----|---|--------|-----|-----------|------|
| FP16 | 192 | 576 | 48 | 112 | 17 | 8 | ~953 |
| BF16 | 192 | 576 | 48 | 112 | 17 | 8 + const | ~1100 |
| FP32 | 96 | 48 | 48 | — | 16 | — | ~208 |
| 空 | 1 | — | — | — | — | — | 1 |
| **总计** | | | | | | | | **~2300** |

## 4. arch35 (950PR) Tiling Key Bit 编码

| Bit 位置 | 宽度 | 字段名 | 值域 | 说明 |
|----------|------|--------|------|------|
| 1-0 | 2 | KernelTypeKey | 0, 1 | — |
| 3-2 | 2 | ImplMode | 0-2 | — |
| 7-4 | 4 | Layout | 0-4 | — |
| 17-8 | 10 | S1TemplateType | 0,16,64,128,256 | 实际维度值 |
| 27-18 | 10 | S2TemplateType | 0,16,32,64,128,256,512 | — |
| 39-28 | 12 | DTemplateType | 0,16,32,48,64,80,96,128,160,192,256,768 | — |
| 51-40 | 12 | DvTemplateType | 同上 或 0(=D) | — |
| 55-52 | 4 | PseMode | 0-4, 9 | 5种PSE模式 + NONE |
| 56 | 1 | HasAtten | 0, 1 | — |
| 57 | 1 | HasDrop | 0, 1 | — |
| 58 | 1 | HasRope | 0, 1 | — |
| 60-59 | 2 | OutDtype | 0-2 | — |
| 61 | 1 | Regbase | 0, 1 | arch35 标记 |

## 5. arch35 模板选择（ASCENDC_TPL_SEL 入口）

| 入口块 | dtype | Layout | S1×S2 | D/Dv | ImplMode | 特殊约束 |
|--------|-------|--------|-------|------|----------|---------|
| 1 | BF16/FP16 | SBND/BNSD/TND | 64×128 | 192/128 | HIGH_PREC | Dv显式指定 |
| 2 | BF16/FP16 | SBND/BNSD/TND | 64×128 | 64-256 | HIGH_PREC | Dv=D |
| 3 | BF16/FP16 | SBND/BNSD/TND | 64×256 | 192/128 | HIGH_PREC | Dv显式指定, 无Pse |
| 4 | BF16/FP16 | SBND/BNSD/TND | 64×256 | 64-256 | HIGH_PREC | Dv=D, 无Pse |
| 5 | BF16/FP16 | SBND/BNSD/TND | 128×128 | 192/128 | HIGH_PREC | Dv显式指定 |
| 6 | BF16/FP16 | SBND/BNSD/TND | 128×128 | 64-768 | HIGH_PREC | Dv=D |
| 7 | BF16/FP16 | SBND/BNSD/TND | 128×256 | 64 | HIGH_PREC | 无Pse |
| 8-14 | BF16/FP16 | TND | 类似 | 类似 | HIGH_PREC | TND layout |
| 15-18 | BF16/FP16 | SBND/BNSD/TND | 128×128 | 192/128,64-768 | INVALID_LINE_HP | HasAtten=1 |
| 19-22 | BF16/FP16 | TND | 128×128 | 类似 | INVALID_LINE_HP | TND+HasAtten=1 |
| 23-25 | FP32 | SBND/BNSD/TND | 64×128 | 768 | HIGH_PREC | FP32 D=768 |
| 26-28 | FP32 | SBND/BNSD/TND | 128×128 | 192/128,64-256 | HIGH_PREC | — |
| 29-31 | FP32 | TND | 类似 | 类似 | HIGH_PREC | TND layout |
| 32-33 | FP8_E5M2/E4M3FN/HIFLOAT8 | SBND/BNSD/TND | 128×512 | 64,128 | HIGH_PREC/INVALID_LINE_HP | OutDtype=FP16 |
| 34 | 空 | — | — | — | — | KernelTypeKey=1 |

## 6. arch35 模板数量统计

| dtype | 估计数量 | 说明 |
|-------|---------|------|
| BF16/FP16 | ~5000 | 大量 D/Dv/S1/S2/PseMode 组合 |
| FP32 | ~900 | D=768/192/64-256 等 |
| FP8 | ~50 | D=64/128, S2=512 限定 |
| 空 tensor | 1 | — |
| **总计** | **~6000** | — |

## 7. Triton-Ascend 映射

| CANN 概念 | Triton-Ascend 对应 |
|-----------|-------------------|
| tiling key bit 字段 | `tl.constexpr` meta 参数 |
| ASCENDC_TPL_SEL 编译期展开 | `@triton.autotune` 或 `@triton.Config` |
| if constexpr dispatch | 单 kernel 内 `if IS_CAUSAL:` 等 |
| UB0/UB1/Block 切分轴 | `BLOCK_M` / `BLOCK_N` / grid 维度 |
| DataType bit | dtype 参数 + 类型模板 |
| Layout bit | stride 计算逻辑 |
| Sparse bit | `IS_CAUSAL` / mask 逻辑 |
| HasPse/HasAttenMask/HasDropOut | `HAS_PSE` / `HAS_MASK` / `HAS_DROPOUT` constexpr |
| S1/S2/D TemplateType | `BLOCK_M` / `BLOCK_N` / `HEAD_DIM` constexpr |
