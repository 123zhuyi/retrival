---
id: ops-transformer-block-sparse-attention-grad
title: CANN ops-transformer BlockSparseAttentionGrad source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/block_sparse_attention_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer BlockSparseAttentionGrad 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/block_sparse_attention_grad/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（与 [[ops-transformer-block-sparse-attention]] 前向同源同提交）。`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。本算子为 `block_sparse_attention`（前向，代号 RFA）的**反向梯度算子**，共享前向的类 CUTLASS `attn_infra` 模板库，但反向专用件不同，且**不复用**前向的 `Mask2IdxAndCount` CSR 压缩。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/block_sparse_attention_grad/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-22
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单（含行数、role、centrality、关键符号、行号区域）见 `artifacts/block-sparse-attention-grad/source_file_inventory.md`。关键文件 SHA-256：

| 源文件（相对算子目录） | SHA-256 | 关键区域 |
|---|---|---|
| `README.md` | `1f3371af5ab28605c06cb6490311a4b4f2865c598f124e35889539ddc23f4b20` | 反向公式 L19-71；约束 L361-367；产品支持 L5-14 |
| `docs/aclnnBlockSparseAttentionGrad.md` | `e6039ae6b18e9de38982fc7ccbb560cb1d18e640fc31be52b1dd8206e763ad04` | 反向公式 L19-71；参数表 L91-320；产品支持 L5-14 |
| `op_host/block_sparse_attention_grad_def.cpp` | `82f08000735bb87162472e33b3a7686f9e643553e6445486145d2466bc286582` | 输入 L19-62；输出 dq/dk/dv L63-74；属性 L76-82；AICore(910b/910_93) L84-85 |
| `op_host/block_sparse_attention_grad_infershape.cpp` | `2bf5429bfecd36cd254a17af47a326b268ddaabd1c5cb1bce6eb3ee7334d8489` | dq=query/dk=key/dv=value 形状直拷 L82-84 |
| `op_host/block_sparse_attention_grad_tiling.h` | `72b67854ff079a2d2078f6e21d958afbc5f0bde442b093b2537af63c252d3dbd` | `BlockSparseAttentionGradTilingData` L22-77；`SoftMaxTiling` 嵌套 L77 |
| `op_host/block_sparse_attention_grad_tiling.cpp` | `77c2a1d037e3a82df986014c1380da12e6f22c12fcce3b421bb4fa4cdaf40647` | `GenerateTilingKey`(4 key) L497-514；`CalculateTaskSplit` L369-420；`CalculateWorkSpace` L422-447 |
| `op_host/op_api/aclnn_block_sparse_attention_grad.cpp` | `178b25f5b840e80bd96a842a61c87bc0a1475febdfeec5242235eca558a43042` | `ValidateParams` L108-188；GetWorkspaceSize L236-307 |
| `op_kernel/block_sparse_attention_grad.cpp` | `75f8e2a15551be3e365646eb9f8c38bd7624d4412d51c5dfb875b73123aaaf12` | 入口 L24-38；4 tiling key 分派 L44-60 |
| `op_kernel/block_sparse_attention_grad_kernel.h` | `bf48e66d0f8b102f8bb032cc4e12c44e9aa867913993d0a3f467925ac2aaf63e` | `BlockSparseAttentionGradKernel` L61；cube `operator()` L291-513；mask 门控扫描 L386-389,604-607；3 flag L39-41 |
| `op_kernel/block_sparse_attention_grad_interface.h` | `24fc40ced2d3381f25175e67b328da3687e09f3e9c71455f74217bf2a0db178f` | 3 cube GEMM 分组 L48/63/78；模板实例化 L123-125 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_simply_softmax.hpp` | `6ef90496ebd157bcbedbd73a0c79cc701a13dd880f5f671c6659a02ea7e93bcb` | 重算 P=exp(S·scale−Lse) L335-393；dS=P·(dP−D) L402-444 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_softmaxgrad.hpp` | `64dd0fe2db2737573e861ab5d79ff78d9381d9408198aa1c8431425e6e974465` | D=Σ(dout·out) L208-297 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_post.hpp` | `a5ddfb5d4b7f709fe592cfdc45042a10780a1ca83b11c56eba74361cc57641ea` | rescale(dQ/dK*=scale)+cast+写回 L335-411 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_fag_pre.hpp` | `d67dbade6e934bd1eaff5a274edaa08ad5640c7c90c68e84a2f8d71c54f2fb89` | dq/dk/dv workspace 清零 L140-166 |
| `op_kernel/attn_infra/gemm/block/block_mmad_bsag2.hpp` | `ae0bd2b7fd0478eb986e6f24100de725d883406fe099a25ba07dc11c66f6910e` | 共享 BlockMmad（3 组 cube GEMM）；AtomicAdd 累加 L149-164 |

## 3. 关键代码区域摘要

### 3.1 反向计算公式（docs L19-71；README L19-71）
前向 `S=Mask(scale·Q·K_sparse^T)`，`P=Softmax(S)`，`Out=P·V`。反向：
- `softmax_grad = softmaxGrad(dOut, attentionOut)`
- `dP = dOut·V^T`
- `dS = P·(dP − softmax_grad)`
- `dV = P^T·dOut`
- `dQ = (dS·K)·scale`
- `dK = (dS^T·Q)·scale`

**P 在反向重算**：`P=exp(S·scale−Lse)`，用前向保存的 `softmaxLse`（`block_epilogue_simply_softmax.hpp` L335-393）；`attentionOut` 作为 `softmaxGrad` 输入。S、P 均不从前向传入，反向重算。

### 3.2 算子定义（def.cpp L19-85）
- 输入：dout/query/key/value/attentionOut/softmaxLse（必选，FP16/BF16；softmaxLse 为 FLOAT）、blockSparseMask（当前必选；def=UINT8，README=BOOL，kernel=bool，三处不一致 [needs_verification]）、attenMask（必传 nullptr）、blockShape（INT64 `[X,Y]`）、actualSeqLengths/actualSeqLengthsKv（TND 必传）。
- 输出：dq/dk/dv（FP16/BF16）。
- 属性：qInputLayout/kvInputLayout（TND/BNSD，须一致）、numKeyValueHeads、maskType（仅 0）、scaleValue、preTokens/nextTokens（仅 INT_MAX）。**无** innerPrecise、**无** softmaxLseFlag（前向专用）。
- AICore config：`ascend910b` + `ascend910_93`（同前向）。

### 3.3 块稀疏 mask 处理（与前向不同，kernel.h L386-389,604-607）
反向**不复用**前向的 `Mask2IdxAndCount` CSR 压缩（无 selectIdx/selectNumIdx）。直接读原始 mask（`GlobalTensor<bool>`），cube 与 vec 核**均**遍历所有 `kvBlockNum` 块，`if (gBlcokSpaseMask.GetValue(maskOffset))` 为真才计算，否则跳过--**mask 门控稠密扫描**（branch-on-mask），非 CSR 索引 gather。mask offset = `curBatchIdx*batchBlocks + curHeadIdx*headBlocks + curQBlockIdx*kvBlockNum + idx`（L388,606）。

### 3.4 流水（kernel.h L291-531，4 阶段 cube↔vec）
4 阶段顺序流水（1-deep `preTaskInfo` delay），3 个 cross-core flag（`CUBE2VEC=7`/`VEC2CUBE=8`/`CUBE2POST=9`，L39-41）：

| 阶段 | 单元 | 入口 | 操作 |
|---|---|---|---|
| Stage1 cube | AIC | kernel.h L405-422 | S=Q·K^T + dP=dOut·V^T（blockMmad1）-> set CUBE2VEC |
| Stage2 vec | AIV | kernel.h L597-675 | wait CUBE2VEC；D=Σ(dY·O)（SoftmaxGrad）；重算 P；dS=P·(dP−D) -> set VEC2CUBE |
| Stage3 cube | AIC | kernel.h L423-461,476-511 | wait VEC2CUBE；dQ=dS·K、dK=dS^T·Q、dV=P^T·dOut（blockMmad2/3，**AtomicAdd** 累加）-> set CUBE2POST |
| Stage4 vec | AIV | kernel.h L515-531 | wait CUBE2POST + SyncAll；rescale(dQ/dK*=scale)+cast+写回（BlockPost） |

dQ/dK/dV 跨 KV 块累加（Cube2/Cube3 `isAtomicAdd=true`，`block_mmad_bsag2.hpp` L159-162）；dQ/dK 写回时乘 scale，dV 不乘（`block_epilogue_post.hpp` L371-374）。3 组 cube GEMM 复用同一 `SBSAG2` 策略。

### 3.5 tiling 与分核（tiling.cpp）
- **4 个 tiling key**（dtype×layout：0/1/10/11，L497-514），抛弃前向 12-key RFA 方案；无 innerPrecise/lse/maskType/pagedCache 轴。
- 核间仅在 (Q-block × head) 切分（`blockDim=min(aicNum, totalTaskNum)`，L418），无 FlashDecoding、无 S2 切分。
- 每核任务描述子数组 `beginBatch/beginHead/beginQSeqOffset/preQSeqLengths/preKVSeqLengths[64]`（与前向 sub-core 分区不同）。
- GQA：`groupSize=numHeads/kvHeads`（L217-221）。
- 基本块 `BASIC_BLOCK_SIZE=128`、`basicQBlockSize=basicKVBlockSize=128`；非对齐 ceilDiv。
- workspace（FP32）：`[gS(重算S) | gDp(dP) | gDq | gDk | gDv | sftmg(softmax-grad)]`（L422-447），**无** selectIdx/selectNumIdx。

### 3.6 产品支持
仅训练：Atlas A2 训练 √、Atlas A3 训练 √；推理系列全 ×（docs L5-14）。前向 A2/A3 训练+推理均 √，反向训练专用。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用：
- `wiki/kernels/block-sparse-attention-grad.md`
- `artifacts/block-sparse-attention-grad/`（source_file_inventory.md、tiling_dispatch_tables.md、migration-table.md、block-sparse-attention-grad-triton-skeleton.py）

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数数值、UB/L1 占用、mask 门控 vs CSR 的稀疏度取舍、AtomicAdd 数值精度、mask dtype 不一致等需结合硬件环境进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
