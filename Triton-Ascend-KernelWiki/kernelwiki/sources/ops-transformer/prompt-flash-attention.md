---
id: ops-transformer-prompt-flash-attention
title: CANN ops-transformer PromptFlashAttention source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/prompt_flash_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer PromptFlashAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/prompt_flash_attention/` 目录抽取的不可变来源信息，用于为相关 wiki 页提供来源追溯。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（与 [[ops-transformer-flash-attention-score]] 同源同提交）。`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/prompt_flash_attention/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-29
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单（含行数、role、centrality、关键符号、行号区域）见 `artifacts/prompt-flash-attention/source_file_inventory.md`。关键文件 SHA-256：

| 源文件（相对算子目录） | SHA-256 |
|---|---|
| `README.md` | `e5524e4e192fc86bb89de37235fe181583c8c94003f1f8287f76ae8369e0a695` |
| `docs/PFA算子设计介绍.md` | `6a5e2b076a942d33a43325df4b63cf9132c7f8b88c547c8da0a5c49d6ec5fcee` |
| `docs/aclnnPromptFlashAttention.md` | `18a6df2ae0500cd1046833280902b3286dfab45fcd57d07d25387bec74ed6915` |
| `docs/aclnnPromptFlashAttentionV2.md` | `7dea6ccc395ce3e3e6f459aa135ccfb1811588b3055d2cb132077358088a1ab0` |
| `docs/aclnnPromptFlashAttentionV3.md` | `b34723da9fbf6e6908551d1e89264475fe845caed6f43af7db7f612f6c90f676` |
| `op_host/prompt_flash_attention_def.cpp` | `e34dcb4534aa2886d99fa903e8a3d29fc2fd462a4aa0f248335a49b0491b9da7` |
| `op_host/prompt_flash_attention_infershape.cpp` | `67258a42a7bb601625cc7cc70aa803404b11a9f714c86cd780abb9810c4ed8e1` |
| `op_host/prompt_flash_attention_tiling.cpp` | `7a991bdbf31a81e198b2752e48e7a0da9affbb5988e89301e77e2b689fe7ea82` |
| `op_host/prompt_flash_attention_tiling.h` | `2a471558f53b9ff0f265f97b1b834fbb45efee127d9a6443a72479b3d5e356ee` |
| `op_host/prompt_flash_attention_tiling_const.h` | `f32ec91ba467337db05e642a04b0ebcbe2520f024b82051debd25060cf10764b` |
| `op_kernel/prompt_flash_attention.cpp` | `74f2f6bd35d6f1adff2cea9da8063de0c9a7377231527e66c714c6107fd6ba54` |
| `op_kernel/prompt_flash_attention_tilingkey.h` | `c5b7ffa59784b248091789cb3f30385aa2d12511e80400e9faf908279737445b` |
| `op_kernel/arch35/prompt_flash_attention_template_tiling_key.h` | `24c1b73a34e93cb1c1c60a4a3617b64a1f0ee497bf6cad5575debd3efb43b006` |
| `op_kernel/arch35/prompt_flash_attention_entry_regbase.h` | `a91cbe229e8ca59936c4d5ddd635589800a782f9330a8a0356e62cc8be73ddc8` |
| `op_kernel/arch35/service/prompt_flash_attention_mm1.h` | `e8eca5c364f44f68632888c2dcbdef2a49e636bb7656b8113c2078b720ac1123` |
| `op_kernel/arch35/service/prompt_flash_attention_vector1.h` | `5ce1375116f60c8de3c51af2e27969ba3daf781ee149314e7501f587e2a0745a` |
| `op_kernel/arch35/vf_softmaxflashv3.h` | `b66741db757a2097c4d9cf286f4c10c52db04368903b696dfd4e7110b9b90aff` |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源：README.md L24-32）
`Attention(Q,K,V) = Softmax(QK^T / √d) · V`。PFA 为全量推理（prefill）场景。

### 3.2 在线 FlashSoftmax 流程（来源：PFA算子设计介绍.md L9-22）
Q·K^T -> +pse -> ×scale -> atten_mask select(true->负极小) -> FlashSoftmax(对 Skv 切分在线刷新) -> ·V -> ÷softmax_sum 写回。rescale：`exp[i] = e^(max[i-1]-max[i])`，`MM[PV][i] = MM[PV][i-1]·exp[i] + MM_new[PV][i]`。

### 3.3 算子定义（来源：prompt_flash_attention_def.cpp L26-103）
- 输入：query/key/value（REQUIRED，FP16/BF16/INT8），pse_shift/atten_mask/actual_seq_lengths(_kv)/deq_scale1/quant_scale1/deq_scale2/quant_scale2/quant_offset2（OPTIONAL）。
- 输出：attention_out（REQUIRED）。kernel 入口（prompt_flash_attention.cpp L36）另输出 `softmaxLse`（推理场景）。
- 属性：num_heads（必需）、scale_value(默认1.0)、pre_tokens(默认214748647)、next_tokens(默认0)、input_layout(默认BSH)、num_key_value_heads(默认0)、sparse_mode(默认0)、inner_precise(默认1)。
- AICore config：ascend910b、ascend910_93（arch32，opFile `prompt_flash_attention`）；ascend950（arch35，opFile `prompt_flash_attention_apt`）；ascend310p（仅 FP16）。

### 3.4 架构与 tiling 分发（来源：tiling.cpp L429/433/1469/3920；tilingkey.h；template_tiling_key_enum.h）
- arch32（A2/A3）：十进制 tilingkey（~130 宏，组装 dtype×layout×precision×API×feature）。
- arch35（950）：结构化枚举模板分发（Layout/S1/S2/D/DV/ImplMode/PseMode/QuantMode/Mask/MatMulType/SplitCoreMode + 27 个 ConfigValue）。
- MLA 触发：`headSize != headSizeV` 或 `headSize > 256`（tiling.cpp L2848）。
- splitCoreMode：`actualSeqLength >= 3072 && b*n*sOuter >= coreNum` -> SPLIT_ONEN_CUBE，否则 SPLIT_NBS_CUBE（tiling.cpp L926-932）。

### 3.5 CV tiling 分离与流水（来源：PFA算子设计介绍.md L55-59/119-138）
Cube 基本块 128×128，Vector 8×1024，nRatio=8（1:16 CV 配比）。四阶段 CV 流水 Bmm1->Vec1->Bmm2->Vec2；V 侧 double buffer；Cube 双发。

### 3.6 kernel 入口（来源：prompt_flash_attention.cpp L25-59）
`prompt_flash_attention_FIAS`（全量入参 V2/V3：含 blocktable/antiquant/sharedPrefix/queryRope/keyRope/dequantScaleQuery/learnableSink/softmaxLse）转发 arch32；`prompt_flash_attention`（V1 兼容，扩展入参传 nullptr）。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用：
- `wiki/kernels/prompt-flash-attention.md`
- `artifacts/prompt-flash-attention/`（source_file_inventory.md、tiling_dispatch_tables.md、pipeline-diagram.md、migration-table.md、prompt-flash-attention-triton-skeleton.py）

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数具体数值、UB/L1 占用 bound、FP8/FP4 精度边界、softmaxLse 精确公式、KV-cache flashupdate 写回时机等需结合硬件环境进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
