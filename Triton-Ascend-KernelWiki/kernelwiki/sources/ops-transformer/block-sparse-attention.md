---
id: ops-transformer-block-sparse-attention
title: CANN ops-transformer BlockSparseAttention source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/block_sparse_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer BlockSparseAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/block_sparse_attention/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（与 [[ops-transformer-flash-attention-score]]、[[ops-transformer-sparse-flash-attention]] 同源同提交）。`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。本算子内部代号 **RFA = Rain Fusion Attention**（旧名），`aclnn_rain_fusion_attention.h` 为公开 `aclnnBlockSparseAttention*` API 的别名头；反向算子 `block_sparse_attention_grad` 为独立算子（同提交、共享 `attn_infra` 模板库），本页不覆盖。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/block_sparse_attention/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-22
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单（含行数、role、centrality、关键符号、行号区域）见 `artifacts/block-sparse-attention/source_file_inventory.md`。关键文件 SHA-256：

| 源文件（相对算子目录） | SHA-256 | 关键区域 |
|---|---|---|
| `README.md` | `335c73fac8db12474a9ec195dbedfc1a0d8a5c03510eda35afe42dde4f84c385` | 功能/约束 L156-166；稀疏 pattern L123-146；A2 硬件 L12-13 |
| `docs/aclnnBlockSparseAttention.md` | `9f556f91eb343cdd6e6a8e76ef702b3a541bd5069f6919105c82521e7773c6d2` | 公式 L18-22；参数表 L77-401；约束 L487-509；产品支持 L5-12 |
| `op_host/block_sparse_attention_def.cpp` | `577fa85542a97093eef8e2c936d82d511b2d511ea83e6aa598eb94f7cb1624f3` | 输入 L19-54；输出 L55-62；属性 L64-73；AICore config(ascend910b/910_93) L75-76 |
| `op_host/block_sparse_attention_infershape.cpp` | `561252f1abb376f82d47cddf4b68628414e99b7ebbdb24e81c87895f957d0cfb` | attentionOut=query L105；softmaxLse TND[T,N,1]/BNSD[B,N,S,1] L183-201 |
| `op_host/block_sparse_attention_tiling.h` | `88949353e01be85dd2160089ab1d94957ddf30b1975388b90ba8602a86d1eef8` | `BlockSparseAttentionTilingData` L22-72；枚举 L101-117 |
| `op_host/block_sparse_attention_tiling.cpp` | `503b9b85b05223d5b4088c847ebb99afad79f9d717139d8a0f332cc86db3c390` | `GenerateTilingKey` L683-748；`CalculateTaskSplit` L557-603；`CalculateWorkSpace` L605-630；`CheckSparsePattern` L390-437 |
| `op_host/op_api/aclnn_block_sparse_attention.cpp` | `d6e6c59c552e3fc705d8180441ae2b295953960210b6cbbd9ce6da25f828a2c6` | GetWorkspaceSize 校验 L65-205；aclnnGetWorkspaceSize L214-289 |
| `op_kernel/block_sparse_attention.cpp` | `8088a81373fe3eb684ab0341aa355026c097c7730f49054624cc703f89fa9163` | 入口 L16-19；`KERNEL_TYPE_MIX_AIC_1_2` L23；tiling key 分派 L29-94 |
| `op_kernel/block_sparse_attention_kernel.h` | `509d1101498ad7b4c6284cd193c9d492688b5a8a1b62c304765b6387cc43ef71` | `Mask2IdxAndCount` L93-248；主循环 L447-731；cross-core flag L782-784 |
| `op_kernel/block_sparse_attention_tilingkey.h` | `28366ca89f50b2ff13d44e0898cb388fa5e709437cc3b8dc5e521efdf3dcf41d` | `RFA_BASE_TILING` L28；12 个 key 常量 L31-65 |
| `op_kernel/block_sparse_attention_interface.cpp` | `851ffe7c37587cc62f03d9108bdc45ffb8269192c76f2c4f05f11fb34f3a5a4a` | tile shape/分派策略 L68-90；模板实例化 L99-110 |
| `op_kernel/kernel_common.hpp` | `8310a58a2145e39d0ef628c300c18f8c79545a1788f9a860dda220b207c262f4` | 同步事件 id L34-37；块常量 L39-84；`FASparseKernelParams` 别名 L175-176 |
| `op_kernel/attn_infra/gemm/block/block_mmad_qk.hpp` | `918e6844391bee7fb83d37b74b53e20b89d5196443b8527c68ebc91a25b8df46` | `BlockMmadQK::operator()` L152-227：沿 selectIdx 稀疏遍历 KV |
| `op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp` | `91c1de54e87d4071747bc098c4ff5ae5950d7c980632aee5c63c8b95193680bf` | `BlockMmadPV::operator()` L121-186：稀疏遍历加载 V |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax.hpp` | `1caea0f2c30a0ff962b397103c7c4b0b82d48bfc9eb987653fbc642d05975f77` | 在线 softmax FP32：running max/sum L280-392 |

## 3. 关键代码区域摘要

### 3.1 计算公式（docs L18-22）
稀疏块大小 `blockShapeX × blockShapeY`，`blockSparseMask` 指定每个 Q 块选择的 KV 块：
`attentionOut = Softmax(scale · query · key_sparse^T + atten_mask) · value_sparse`。`blockSparseMask` shape `[batch, headNum, ceilDiv(maxQS, blockShapeX), ceilDiv(maxKvS, blockShapeY)]`，INT8（1=参与，0=跳过）。

### 3.2 算子定义（def.cpp L19-76）
- 输入：query/key/value（必选，FP16/BF16，TND `[T,N,D]` / BNSD `[B,N,S,D]`，D∈{64,128}）、blockSparseMask（当前必选，INT8）、attenMask（必传 nullptr）、blockShape（INT64 `[X,Y]`，Y%128==0）、actualSeqLengths/actualSeqLengthsKv（TND 必传）、blockTable（必传 nullptr）。
- 输出：attentionOut（同 query）、softmaxLse（FLOAT，softmaxLseFlag=1 时，TND `[T,N,1]` / BNSD `[B,N,S,1]`）。
- 属性：qInputLayout/kvInputLayout（TND/BNSD，须一致）、numKeyValueHeads、maskType（仅 0）、scaleValue、innerPrecise（0=fp32/1=fp16 softmax；BF16 仅 0）、blockSize（仅 0）、preTokens/nextTokens（仅 INT_MAX）、softmaxLseFlag。
- AICore config：`ascend910b` + `ascend910_93`（A2/A3）。

### 3.3 块稀疏 mask 压缩（kernel.h L93-248，核心优化）
非"遍历所有 KV 块置 -inf"。vector 核 `Mask2IdxAndCount` 一次性把 INT8 mask 压缩为 CSR 式 `selectIdx`（选中 KV-block 索引，int32）+ `selectNumIdx`（每行选中数，int32）：DataCopyPad int8 -> Cast -> Compare≥1 -> GatherMask 紧凑化，ping-pong。cube 核 GEMM（`block_mmad_qk.hpp` L152-227、`block_mmad_pv.hpp` L121-186）沿 `selectIdx` gather 加载 K/V，未选中块完全不加载；`curSelectNum==0` 时整 Q-block 跳过（kernel.h L510-514）。详见 [[technique-block-sparse-mask-gather]]。

### 3.4 流水（kernel.h L596-731）
4 阶段 cube↔vector 流水（同 FlashAttention 骨架，详见 [[technique-flash-attention-pipeline]]），3 路循环 buffer（`PRE_LAUNCH=2`）：

| 阶段 | 单元 | 入口 | 操作 |
|---|---|---|---|
| Stage1 QK(bmm1) | AIC | kernel.h L616-633 | Q·K^T（沿 selectIdx gather K）-> set qkReady |
| Stage2 在线 softmax | AIV | kernel.h L642-652 | wait qkReady；running max/sum + exp rescale -> set softmaxReady |
| Stage3 PV(bmm2) | AIC | kernel.h L678-697 | wait softmaxReady；P·V（沿 selectIdx gather V）-> set pvReady |
| Stage4 rescale O/写回 | AIV | kernel.h L717-731 | wait pvReady；O/=l_i；可选写 LSE |

3 个 cross-core flag（qkReady/softmaxReady/pvReady，id 1/2/3，`kernel_common.hpp` L34-37）为 CANN 专有。

### 3.5 tiling 与分核（tiling.cpp）
- 12 个 tiling key（`tilingkey.h` L31-65），按 dtype×Q/KV-layout×softmax-precision×softmaxLseFlag 分派；D/blockShape 不进 key。
- 核间仅在 (Q-block × head) 轴切分（`blockDim=min(aicNum, totalTaskNum)`，L601），无 FlashDecoding、无 S2 核间切分。
- GQA：`groupSize=numHeads/kvHeads`（L572），N1≥N2 && N1%N2==0。
- 基本块 `BASIC_BLOCK_SIZE=128`、`Q_TILE_CEIL=128`、blockShape 默认 `[128,128]`；非对齐 ceilDiv。
- workspace 三缓冲：`[gS|gP|gOTmp|gOUpdate|gSelectNumIdx|gSelectIdx]` + sync buffer（L605-630）。

### 3.6 任务类型与架构
`KERNEL_TYPE_MIX_AIC_1_2`（AIC:AIV=1:2，`block_sparse_attention.cpp` L23）；`attn_infra/arch/arch.hpp` 仅 AtlasA2（C220）。产品支持 A2 √ / A3 √ / 其余 ×（docs L5-12）。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用：
- `wiki/kernels/block-sparse-attention.md`
- `wiki/techniques/block-sparse-mask-gather.md`
- `artifacts/block-sparse-attention/`（source_file_inventory.md、tiling_dispatch_tables.md、migration-table.md、block-sparse-attention-triton-skeleton.py）

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数数值、UB/L1 占用、mask 压缩并行策略、innerPrecise 精度边界等需结合硬件环境进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
