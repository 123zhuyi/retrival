---
id: ops-transformer-nsa-compress-attention-infer
title: CANN ops-transformer NsaCompressAttentionInfer source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/nsa_compress_attention_infer/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer NsaCompressAttentionInfer 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/nsa_compress_attention_infer/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。本算子为 NSA（Native Sparse Attention）推理流水线的第一阶段——压缩注意力计算 + 重要性分数 + topK 索引选择。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/nsa_compress_attention_infer/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-22
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单见 `artifacts/nsa-compress-attention-infer/source_file_inventory.md`。关键文件 SHA-256：

| 源文件 | SHA-256 | 关键区域 |
|---|---|---|
| `docs/aclnnNsaCompressAttentionInfer.md` | `6a933afd5b6a36d2...` | NSA 公式 L22-28；参数表 L77-353；约束 L436-442 |
| `op_host/nsa_compress_attention_infer_def.cpp` | `d5b1a47b5a878cc3...` | 输入 L19-56；输出 L57-63；属性 L64-86；AICore L88-89 |
| `op_host/nsa_compress_attention_infer_tiling.cpp` | `d5f04c1f900fbb48...` | Tiling L457-558；AssignTask L327-445 |
| `op_kernel/nsa_compress_attention_infer.h` | `3419e3da635d722d...` | Aic L40-689；Aiv L690-1565；ProcessMm1 L517-559；ProcessTopK L1479-1526 |
| `op_kernel/ncai_public_define.h` | `ce95206e21126d8b...` | NCAIType L46-52；sync event id L34-36 |

## 3. 关键代码区域摘要

### 3.1 NSA 公式（docs L22-28）
`P_cmp=Softmax(scale·Q·K^T)` → `attentionOut=P_cmp·V` → `P_slc[j]=sum over sliding window of P_cmp` → `topKIndices=topk(P_slc')`。

### 3.2 算子定义（def.cpp L19-89）
输入：query（必选，FP16/BF16）、key/value（3D PagedAttention 格式）、blockTable（必选）、actualCmpKvSeqLen（必选）；输出：output、topKOutput（INT32）。属性：NSA 特有 selectBlockSize/Count、compressBlockSize/Stride、pageBlockSize、sparseMode 等。AICore：ascend910b + ascend910_93。

### 3.3 kernel 流水
Cube（AIC）：ProcessMm1（QK^T，blockTable 映射）→ sync → wait softmax → ProcessMm2（PV）。
Vector（AIV）：wait QK^T → ProcessSoftmax（含 mask）→ sync → ProcessImportanceScore（滑动窗口卷积）→ ProcessTopK（输出 int32 索引）。

### 3.4 tiling
2 key（0=TND，1=BSND）。核间切分：B×N2×Q-S。workspace：mm1Res/ mm2In/scoreIn/topKIn 双缓冲。

### 3.5 产品支持
A2 训练+推理 √、A3 训练+推理 √；其余 ×。

## 4. 衍生引用

- `wiki/kernels/nsa-compress-attention-infer.md`
- `artifacts/nsa-compress-attention-infer/`

## 5. 复现性说明

本页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。