---
id: ops-transformer-nsa-compress-attention
title: CANN ops-transformer NsaCompressAttention source snapshot
type: source
repo: cann/ops-transformer
path: attention/nsa_compress_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer NsaCompressAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/nsa_compress_attention/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。本仓库为本地制品目录，`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。

NSA（Native Sparse Attention，arxiv 2502.11089）的压缩注意力阶段，用于选择 topk 索引供下游 `nsa_selected_attention` 使用。相关兄弟算子：`nsa_compress`（压缩）、`nsa_selected_attention`（选择注意力）、`nsa_compress_attention_infer`（推理变体）。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/nsa_compress_attention/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 远端: `https://gitcode.com/cann/ops-transformer.git`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|------|------|------|
| `attention/nsa_compress_attention/README.md` | `d2ac7c7a6cc73a99c5248203acb08e1bde58d4e144564c10f1bdc54695de6737` | 算子总览、NSA公式、I/O参数、约束 |
| `attention/nsa_compress_attention/docs/aclnnNsaCompressAttention.md` | `52d051abb75fc8d1e6dd2cf9832aba414b8e43350d94eeb30ec70ae5ce8be433` | aclnn接口文档 |
| `attention/nsa_compress_attention/op_host/nsa_compress_attention_def.cpp` | `6456c8eead125ae8fb5322a58ffc4f42c95465679dc0c3e26a4cba0f6be545f5` | 算子定义（I/O dtype/layout、属性 compressBlockSize/compressStride/selectBlockSize/selectBlockCount、AICore配置 ascend910b/910_93）|
| `attention/nsa_compress_attention/op_host/nsa_compress_attention_tiling_general.cpp` | `db509e448b38d46366345e8e343f14f699db0c8dcf4685d92ca0329793f1c76b` | tiling基类 + VarLen派生类（CalcS1S2BasicBlock、SetBmm1/2TilingInput、SetMultiCoreParams、SetImportanceScoreParams、TopkTiling）|
| `attention/nsa_compress_attention/op_kernel/nsa_compress_attention.cpp` | `2f49a99de3e8df5dff966219861c0da898a86ac610d9c105b16e3b7632714a14` | kernel入口、tiling key分发（8分支：FP16/BF16 × hasAtten/hasTopkMask）|
| `attention/nsa_compress_attention/op_kernel/nsa_compress_attention_s1s2_bn2gs1_sab.h` | `02e04a7f55450af7a3a89a564af6d3336474f1c650213031f8234e862045ddb9` | 泛化kernel模板（NsaCompressAttentionS1s2Bn2gs1SameAB：Process、IterateBmm1、ProcessVec1、ProcessVec2、IterateBmm2、TopkCompute）|
| `attention/nsa_compress_attention/op_kernel/nsa_compress_attention_var_len_score_sab.h` | `b36a1357fdbd039864e68fd720cfe3a4d267a3cbadca1f3be786e7a505a8cfac` | varlen kernel模板变体 |
| `attention/nsa_compress_attention/op_host/nsa_compress_attention_tiling.h` | `17c276f9dc719c9a0457a40ea3dc7182cb1908510f599e3ba7e044ff6ef4f039` | TilingData结构体（inputParams/coreParams/multiCoreParams/bmm1/2TilingData/softmaxTilingData/importanceScoreParams/topkTilingData）|
| `attention/nsa_compress_attention/op_host/nsa_compress_attention_tiling_common.h` | `bf21295bca6dd634b904c225df0cddd091db768f3d9434229bb5c145ebc893ba` | CompileInfo（aivNum/aicNum/ubSize/l1Size/l0cSize/l2CacheSize）|

## 3. 关键代码区域摘要

### 3.1 NSA 压缩注意力公式（来源: README.md L16-38）

compressBlockSize=l, selectBlockSize=l', compressStride=d:

1. `P_cmp = Softmax(query * key^T)`
2. `attention_out = Softmax(mask(scale * query * key^T, atten_mask)) * value`
3. `P_slc[j] = sum_{m=0}^{l'/d-1} sum_{n=0}^{l/d-1} P_cmp[l'/d * j - m - n]`  （滑动窗口）
4. `P_slc' = sum_{h=1}^{H} P_slc^h`  （跨 head 累加）
5. `P_slc' = topk_mask(P_slc')`
6. `topkIndices = topk(P_slc')`

输出：attention_out（与query形状一致）+ topkIndicesOut（INT32）+ softmax_max/sum（供反向）。

### 3.2 约束（来源: README.md L67-85）

- compressBlockSize(l): 16对齐，≤128；compressStride(d): 16对齐，≤64；满足 l ≥ d
- selectBlockSize(l'): 16对齐，≤128；满足 l' ≥ l，l' % d == 0
- selectBlockCount(n): [1, 32]，≤ min(SelSkv)
- `SelSkv = CeilDiv(CmpSkv, selectBlockSize // compressStride)`
- `CmpSkv ≤ 14000`（UB限制）
- layout仅支持TND；query/key/value dtype一致；G < 128 && 128 % G == 0

### 3.3 Tiling设计（来源: tiling_general.cpp L862-967）

`NsaCompressAttentionVarLenScoreTiling`（注册为最高优先级模板，L967）：
- 基本块：`s1BasicBlock = NSA_BASE_S1G_SIZE / gSize = 128 / gSize`（L873）
- `s1VecBasicBlock = s1BasicBlock / 2`（CV_RATIO=2, L874）
- `s2BasicBlock = alignedS2`（S2不做切分，整块计算，L875）
- 核间切分：`accumS1BlockNum = Σ CeilDiv(actualSeqLenData[i], s1BasicBlock)`，`totalSize = accumS1BlockNum * n2Size`，`coreNum = min(totalSize, aicNum) * 2`（SameAB AIC×2, L924-942）
- workspace：`pWorkspaceSize + impScoreWorkspaceSize + stage1ResWorkspaceSize + 16MB`（L949-958）

### 3.4 Kernel dispatch（来源: kernel L66-113）

tiling key `10000000000000000024UL`，按 dtype（FP16/BF16）× hasAttenMask × hasTopkMask 分8分支。仅1个kernel模板类（`NsaCompressAttentionS1s2Bn2gs1SameAB`）+ 1个varlen变体（`NsaCompressAttentionVarLenScoreSameAB`）。

### 3.5 Importance score + topk（来源: kernel ProcessVec2 L550-774 + TopkCompute L788-908）

- `isM = selectBlockSize / compressStride`，`isN = compressBlockSize / compressStride`，`S2PerScore = isM + isN - 1`
- s2 轴分 `outerLoop` 块，每块做 `DataCopyPad + ConfusionTranspose + Muls + Add + ConfusionTranspose + DataCopy`
- 跨 head reduce g：二路归并 Add 循环（L709-741）
- Topk：`TopK<float, false, false, false, TOPK_NORMAL>` API，选 topk 索引（L878-888）

### 3.6 算子定义的 dtype/layout/属性（来源: def.cpp）

- 输入：query/key/value（FP16/BF16）、atten_mask（BOOL/UINT8）、actual_seq_qlen/cmp_seq_kvlen/sel_seq_kvlen（INT64）、topk_mask（BOOL/UINT8）
- 输出：softmax_max/softmax_sum（FLOAT）、attention_out（FP16/BF16）、topk_indices_out（INT32）
- 属性：scale_value(1.0)、head_num(必需)、input_layout(必需，仅TND)、sparse_mode(0)、compress_block_size(必需)、compress_stride(必需)、select_block_size(必需)、select_block_count(必需)
- AICore配置：ascend910b、ascend910_93（不支持 ascend950）

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/nsa-compress-attention.md`
- `artifacts/nsa-compress-attention/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数具体数值、UB 占用、topk UB 限制、TND 布局偏移等需结合硬件环境进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`，未决项标记 `needs_verification`。本页为 `static_ingest` 静态分析，未运行/未验证/未 benchmark 算子。