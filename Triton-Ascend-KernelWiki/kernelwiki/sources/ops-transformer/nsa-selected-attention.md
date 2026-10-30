---
id: ops-transformer-nsa-selected-attention
title: CANN ops-transformer NsaSelectedAttention source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/nsa_selected_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer NsaSelectedAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/nsa_selected_attention/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。本算子为 NSA（Native Sparse Attention）训练前向的第二阶段，根据 `topKIndices`（来自 `nsa_compress_attention_infer`）gather 选中的 KV 块并执行标准 attention 计算，输出 attentionOut + softmaxMax/Sum（供反向使用）。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/nsa_selected_attention/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-22
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单见 `artifacts/nsa-selected-attention/source_file_inventory.md`。关键文件：

| 源文件 | SHA-256 | 关键区域 |
|---|---|---|
| `docs/aclnnNsaSelectedAttention.md` | `8c41faf664756fdc...` | 公式 L22-28；参数表 L68-353；约束 L358-380；产品支持 L5-12 |
| `op_host/nsa_selected_attention_def.cpp` | `7f0272ba20407201...` | 输入 L19-55；输出 L56-67；属性 L68-79；AICore L88-89 |
| `op_host/nsa_selected_attention_tiling_general.cpp` | `a5272ae0dfba6837...` | `GetTilingKey` 0/1 L565-572；`AnalyzeAttrs` L339-480；`CalcWorkspaceSize` L637-650 |
| `op_kernel/nsa_selected_attention.h` | `d669712408ea1fa2...` | `Process()` L359-392；`SelectAndGatherK` L447-481；`ProcessSoftmax` L524-581；`HandleMask` L621-666 |
| `op_kernel/nsa_selected_attention_common.h` | `bad41d68f8526098...` | `SOFTMAX_REDUCE_SIZE=8` L56；`SparseModeEnum` L85-96 |

## 3. 关键代码区域摘要

### 3.1 公式（docs L22-28）
`selected_key = Gather(key, topk_indices)` → `selected_value = Gather(value, topk_indices)` → `attention_out = Softmax(Mask(scale · Q · selected_key^T, atten_mask)) · selected_value`。输出 attentionOut + softmaxMax/Sum（FLOAT，供反向用）。

### 3.2 算子定义（def.cpp L19-89）
输入：query/key/value（FP16/BF16，仅 TND）、topkIndices（INT32，必选）、attenMask（可选，BOOL/UINT8）、actualSeqQLen/KvLen（TND 必选）。输出：softmaxMax/Sum（FLOAT `[T,N1,8]`）、attentionOut（同 query dtype）。属性：scaleValue、headNum、sparseMode（0/2）、inputLayout（仅 TND）、selectedBlockSize（≤128 16倍数）、selectedBlockCount（[1,128]）。AICore：ascend910b + ascend910_93。

### 3.3 kernel 流水（`.h` L359-392）
外层循环 S1 行 → 内层循环 N2（kvHead）：`GetIndices`（加载 topKIndices）→ `SelectAndGatherK`（gather 选中 K 块）→ `IterateBmm1`（QK^T，与 V gather 重叠）→ `WaitBmm1Result` → `ProcessSoftmax`（在线 softmax + mask + 输出 max/sum）→ `IterateBmm2`（PV）。dQK=192，dV=128。GQA gSize ≤ 32。

### 3.4 tiling（tiling_general.cpp）
2 key（0=无mask，1=causal）。核间切分 S1 行。per-core workspace：selectedK/V + P + softmax 残差。

### 3.5 产品支持（docs L5-12）
仅 A2 训练 √；A2 推理 ×、A3 ×、950PR/950DT ×。

## 4. 衍生引用

- `wiki/kernels/nsa-selected-attention.md`
- `artifacts/nsa-selected-attention/`

## 5. 复现性说明

本页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。