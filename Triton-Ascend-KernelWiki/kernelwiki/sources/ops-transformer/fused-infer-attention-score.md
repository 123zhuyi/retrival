---
id: ops-transformer-fused-infer-attention-score
title: CANN ops-transformer FusedInferAttentionScore source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/fused_infer_attention_score/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer FusedInferAttentionScore 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/fused_infer_attention_score/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。FIA 是推理统一 attention（prefill+decode），核心架构为 `attn_infra/` 分层模板库。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/fused_infer_attention_score/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|------|------|------|
| `attention/fused_infer_attention_score/README.md` | `0d06ea50c9f93df67f90395640a07a0d6efc7b637aafbc37573fb953330a4252` | 算子总览、公式、约束（prefill+decode 统一） |
| `attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV5.md` | `2e2643354f0c54c01292a259be406db977f40efb074606c04c157dd066d1fc06` | V5 aclnn API（主，仅 950） |
| `attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md` | `f9e26375c1a723f60e32ea07f7999c52fb72413ace0b32060076d080dd32978e` | V4 aclnn API（跨架构主力） |
| `attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md` | `3315ac15124a2bdb184ea9dc039ac69960c27d4540203d336ca32027d462c060` | V3 aclnn API |
| `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_def.cpp` | `ad64a9ad1916d8b9b0a1b8722057bf40e01dce2c7982d96062f515a9123cd1ed` | 算子定义（dtype/属性/AICore config） |
| `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp` | `5b066c7476a5c8aeb49fab7fcb9f2035f82d08f15d840a71ae193eb6d0cf7f8a` | tiling 主逻辑（arch 分发/分核/FlashDecode/MLA/PA/GenTilingKey） |
| `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h` | `d41320942fa7173e996e933a54e98fdc99c84c4d0288c60e68fe615e0e4e5245` | tiling 数据结构 |
| `attention/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_v3.cpp` | `602bd85162ad44ffa76a228a1f467aa078e21d4ec6bded4551aab295f9823b30` | arch32 tiling |
| `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp` | `65a9bf28b96e4f411df1e7043e408a5394897e604af0f9745f0b5e987f4413f6` | arch35 tiling |
| `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score.cpp` | `735847454774c89f3d45b2d56827a09e596e9e637c78be5d8f37fdfb93296478` | arch32 kernel 入口 |
| `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_apt.cpp` | `d7adb0cecde4af1f97681656016213208bcb0b78c60d43373212ea632ae93716` | arch35 kernel 入口 |
| `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tiling_key.h` | `d474235a225bef56393f95bbb0dcb494354b2271931f4c3bc3c5ec799e1179b3` | arch35 模板分发（3116 行，if constexpr） |
| `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_tilingkey.h` | `3ad2cf44a9f00c2dc8565e17295961eced28a4926f314c4d8643dd576b7607d0` | arch32 tilingkey 宏 |
| `attention/fused_infer_attention_score/op_kernel/flash_attention_regular.h` | `0cd93ab0a937eba45bdf4e15a4aaf47411d375b5981686ba595bfb9877cbbeda` | prefill regular 模板 |
| `attention/fused_infer_attention_score/op_kernel/flash_attention_regular_decode.h` | `a879b8acc7ef4af9a5948d549335ae705539937ac50c1d63dbd88df0809b9e6f` | decode 模板（含 FlashDecode） |
| `attention/fused_infer_attention_score/op_kernel/attn_infra/gemm/block/block_mmad_qk.hpp` | `a1051a36153eac775470153dd773ae2c5d7e5f03091ddd4a81f2742cf449386f` | gemm Bmm1（Q·K^T） |
| `attention/fused_infer_attention_score/op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp` | `d0f2050d37a98c1ebc4a15f45b04a3d932e3557cfde42a77cdb6ce0ab31edca8` | gemm Bmm2（P·V） |
| `attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` | `9313fbdc40864b002f4d50fd3c05329b007944a6de074b7d8355f2eb24c9df27` | epilogue FlashSoftmax（在线 max/sum/exp） |
| `attention/fused_infer_attention_score/op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o.hpp` | `f62bb85afe9514c361d25c6cdb140e8dc5789d3fcafe17519705061481b35a20` | epilogue rescale ÷softmax_sum + 写回 |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L24-34）

`Attention(Q,K,V) = Softmax(QK^T / √d) · V`。FIA 统一 prefill（Q_S>1）与 decode（Q_S=1），FlashSoftmax 在线刷新，FP32 累加。

### 3.2 attn_infra 分层架构（来源: op_kernel/attn_infra/）

FIA 的 op_kernel 用 header-only C++ 模板库 `attn_infra/` 分层（52 个 `.hpp`），对应 attention 流程：
- `gemm/block/`：`block_mmad_qk`=Bmm1，`block_mmad_pv`=Bmm2（含 `_decode` 变体）。
- `gemm/tile_common/`：Cube 搬运（GM->L1->L0A/L0B->L0C->GM）。
- `epilogue/block/`：`block_epilogue_online_softmax`=FlashSoftmax，`block_epilogue_rescale_o`=写回，`CombineScale`=量化合并。
- `epilogue/tile_common/`：Vector 搬运+elemwise（pse/scale/mask）。
- `layout/`（ND/NZ）、`arch/`（resource/buffer/sync）：CANN 专有，不迁移。

### 3.3 四阶段流程（attn_infra gemm + epilogue 交替）

Bmm1（`block_mmad_qk`）-> Vec1（`epilogue online_softmax`）-> Bmm2（`block_mmad_pv`）-> Vec2（`epilogue rescale_o`）。

### 3.4 prefill vs decode 模板（来源: flash_attention_regular.h / _regular_decode.h）

- `flash_attention_regular.h`（1059 行）：prefill 全量（Q_S>1）。
- `flash_attention_regular_decode.h`（745 行）：decode 增量（Q_S=1，含 FlashDecode）。

### 3.5 架构分发逻辑（来源: fused_infer_attention_score_def.cpp L568-571 / L2048-2050）

`ascend910b`/`ascend910_93` 用 opFile `fused_infer_attention_score`（arch32）；`ascend950` 用 opFile `fused_infer_attention_score_apt`（arch35）。不支持 310P。

### 3.6 FlashDecode（来源: fused_infer_attention_score_tiling.cpp L1426-1439）

decode（Q_S=1）+ BN2 不足 0.4×核数时，S2 外切 SplitKv，各核独立 Attention，lse 加权规约。

### 3.7 V5 新特性（来源: aclnnFusedInferAttentionScoreV5.md L21）

V5 相比 V4 新增 `qStartIdx`/`kvStartIdx`/`pseType`；支持 FP8（HIFLOAT8/FLOAT8_E4M3FN）/FP4（FLOAT4_E2M1）/INT4；分离 KV 反量化（`key/value_antiquant_scale` 替代 `antiquant_scale/offset`）；sharedPrefix、RoPE 解耦（`query_rope`/`key_rope`）、learnableSink、softmaxLse 输出。V5 仅 950（L10-15）。

### 3.8 算子定义的 dtype/layout（来源: fused_infer_attention_score_def.cpp）

- query: FP16/BF16/INT8/HIFLOAT8/FLOAT8_E4M3FN；key/value: 同 + INT4(INT32)/FLOAT4_E2M1；attention_out: FP16/BF16/INT8。
- layout: BSH/BSND/BNSD/BNSD_BSND。
- 属性: num_heads(必需), scale_value, pre_tokens/next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise, block_size, antiquant_mode, softmax_lse_flag, key/value_antiquant_mode, query_quant_mode, pse_type。
- 支持架构: ascend910b, ascend910_93, ascend950（无 310p）。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/fused-infer-attention-score.md`
- `artifacts/fused-infer-attention-score/`（source_file_inventory.md / migration-table.md / tiling_dispatch_tables.md / pipeline-diagram.md / fused-infer-attention-score-triton-skeleton.py / PROVENANCE.yaml）

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。FIA 无 `算子设计介绍.md`，tiling 参数数值、`tiling.cpp` 精确行号、arch35 模板枚举、attn_infra 精确类名/行号、V5 在 A2/A3 支持范围、FP8/FP4/INT4 精度边界需结合源码/硬件进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
