---
id: ops-transformer-flash-attention-score
title: CANN ops-transformer FlashAttentionScore source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/flash_attention_score/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer FlashAttentionScore 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/flash_attention_score/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。本仓库为本地制品目录，`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/flash_attention_score/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-22
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|------|------|------|
| `attention/flash_attention_score/README.md` | `41343354d0207292585b0958240fb743818d05d4fb6da58a03a6ef896213670d` | 算子总览、计算公式、参数说明、约束 |
| `attention/flash_attention_score/docs/FA算子设计介绍.md` | `ddbfdc8fe1efd99e52eab790ad53231887103aaedcd748f059124625e5b61cf3` | 计算流程、tiling设计、流水设计、多模板设计 |
| `attention/flash_attention_score/op_host/flash_attention_score_tiling.cpp` | `c3f2afb269b7196690da73e3282110eb44f4b4953c46f7c433a77bfb1aefdde0` | tiling入口、空tensor处理、架构分发 |
| `attention/flash_attention_score/op_host/flash_attention_score_tiling_common.h` | `cd2e8b8695a0d3fdf82e7a07e18bba7779b6a03fa066fe4f54c5a7b06654afe4` | CompileInfo结构体(UB/L1/L0C/L2/核心数) |
| `attention/flash_attention_score/op_host/flash_attention_score_def.cpp` | `ebcaabe631df622b542f47017c17d820d1f69dc4da6539b6004512ff6295cc8e` | 算子定义(输入输出dtype/layout/属性) |
| `attention/flash_attention_score/op_kernel/flash_attention_score.cpp` | `5b6e7000b469f702a4f05fab9fb0c97ed58729b8e03a3f27eb0c613b1bfbf319` | kernel入口、模板dispatch逻辑 |
| `attention/flash_attention_score/op_kernel/arch32/flash_attention_score_tiling.h` | `6ca76bce8a92fdf63d3c5646dfe6d2af3a9b367a9e6fbf7f994f0a5cee71c8d3` | arch32 tiling数据结构 |
| `attention/flash_attention_score/op_kernel/arch32/flash_attention_score_template_tiling_key.h` | `a8951fea89250f0f815382e9c3b2a05a4c01a12f971985bfe5d133f04062f072` | arch32 tiling key编码(17个bit字段) |
| `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_template_tiling_key.h` | `11dd484c9a4d811275decfff2ccdb24921327ce69d6971e5db0b66536f069078` | arch35 tiling key编码(12个字段) |
| `attention/flash_attention_score/op_kernel/arch35/flash_attention_score_kernel_train.h` | `b8c38401d6b408656af01793e53f61e8dc38634cead79cf4a2c036f997980ae0` | arch35 统一kernel模板 |
| `attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_regbase.h` | `0c73a9bac8d39f251fe93d4edf7faffc854c577c42d76abc17cca8bc8a5db115` | arch35 regbase tiling数据结构 |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L20-32）

- pseType=1 时（先 add 再 mul）:
  `attention_out = Dropout(Softmax(Mask(scale * (pse + (query*d_scale_q) * (key*d_scale_k)^T), atten_mask)), keep_prob) * (value*d_scale_v)`
- pseType≠1 时（先 mul 再 add）:
  `attention_out = Dropout(Softmax(Mask(scale * ((query*d_scale_q) * (key*d_scale_k)^T) + pse), atten_mask), keep_prob) * (value*d_scale_v)`

### 3.2 FlashSoftmax 在线刷新流程（来源: FA算子设计介绍.md L11-18）

1. 按 Skv 轴切分，每次 FlashSoftmax 处理一个 SkvSplit。
2. 从第二次循环开始记录 `exp[i] = e^(max[i-1] - max[i])`。
3. i=0 时 MM[PV] 结果直接存入 ub_attention_out[0]。
4. i≥1 时 `MM[PV][i] = MM[PV][i-1]*exp[i] + MM_new[PV][i]`，存入 ub_attention_out[i]。
5. 最终按行除以 softmax_sum 写回 attention_out。

### 3.3 Pipeline 四阶段入口（来源: FA算子设计介绍.md L22-28）

- `Process`: 分配任务块后总入口
- Cube1 `IterateBmm1`: Q·Kᵀ matmul
- Vector1 `ProcessVec1`: pse+scale+mask+FlashSoftmax
- Cube2 `IterateBmm2`: P·V matmul
- Vector2 `ProcessVec2`: rescale+dropout+output

### 3.4 架构分发逻辑（来源: flash_attention_score_tiling.cpp L345-372）

`TilingFlashAttentionScore` 检测 `NpuArch::DAV_3510`（即 arch35 / 950PR/950DT）走 `TilingRegistryArch::DoTilingImpl`；其余架构走 `TilingRegistryArch` 分发到 arch32 tiling。空 tensor 走独立的 `IsEmptyInputRegbase`/`IsEmptyInput` 路径，tiling key 固定 `FA_EMPTY_TILING_KEY=1`，workspace 预留 100MB。

### 3.5 kernel dispatch 模板参数（来源: flash_attention_score.cpp L380-383）

arch32 kernel 函数签名含 19 个模板参数:
`KernelTypeKey, UB0, UB1, Block, ImplMode, DataType, Layout, Bmm1Format, Bmm2Source, Sparse, BigDoubleBuffer, HasDropOut, HasAttenMask, HasPse, EnableL1Reuse, HasRope, MatmulPolicyType, S1TemplateType, S2TemplateType, dTemplateSize`

通过 `if constexpr` 对 UB0/UB1/Block/Layout 组合做模板分发:
- `UB0==3,UB1==4,Block==9,Layout==4`: TND 模板
- `UB0==3,UB1==9,Block==9,Layout==4`: TND SameAB 模板
- `UB0==3,UB1==9,Block==9,Layout!=4`: SameAB 模板
- `UB0==3,UB1==4,Block==9,Layout!=4`: S1S2 模板
- `UB0==3,UB1==5,Block==9`: S1 模板
- `UB0==9,UB1==9,Block==0`: B 模板

### 3.6 算子定义的 dtype/layout（来源: flash_attention_score_def.cpp）

- query/key/value: FP16/BF16/FLOAT/FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8
- 输入 layout: BSH/SBH/BNSD/BSND/TND
- 属性: scale_value(默认1.0), keep_prob(默认1.0), head_num(必需), input_layout(必需), sparse_mode(默认0), pse_type(默认1), inner_precise(默认0)
- 支持架构: ascend950, ascend910b, ascend910_93

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/flash-attention-score.md`
- `wiki/techniques/flash-attention-tiling.md`
- `wiki/techniques/flash-attention-pipeline.md`
- `wiki/patterns/flash-attention-tiling-key-dispatch.md`
- `artifacts/flash-attention-score/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数具体数值、流水 bound 分析、arch35 UB/L1 使用量等需结合硬件环境进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
