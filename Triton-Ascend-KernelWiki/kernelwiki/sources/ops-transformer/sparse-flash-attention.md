---
id: ops-transformer-sparse-flash-attention
title: CANN ops-transformer SparseFlashAttention source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/sparse_flash_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer SparseFlashAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/sparse_flash_attention/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（与 [[ops-transformer-flash-attention-score]] 同源同提交）。`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。SFA 无 arch32/arch35 子目录（单架构 arch32/A2，MLA 专用），文件直接在 `op_kernel/` 下。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/sparse_flash_attention/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-22
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单（含行数、role、centrality、关键符号、行号区域）见 `artifacts/sparse-flash-attention/source_file_inventory.md`。关键文件 SHA-256：

| 源文件（相对算子目录） | SHA-256 | 关键区域 |
|---|---|---|
| `README.md` | `0502476ff6039fba2a3b6f8c83fa88d85c228f646abaaf74c97fc96b3c3da509` | 功能/公式 L10-21；参数 L29-75；约束 L83-90 |
| `docs/aclnnSparseFlashAttention.md` | `0f496264d160ecd17509291f55603d851eb974bcd9697ab300075e55844a8ea0` | aclnn API 语义 |
| `op_host/sparse_flash_attention_def.cpp` | `e38f748989b4a143b7be9ec11d9f10c2b3046265e2644f8e8601de6794085300` | 输入 L23-67；输出 L68-79；属性 L80-88；AICore config L96-97 |
| `op_host/sparse_flash_attention_tiling.cpp` | `e3fe448c1e9a789804a9548bf703f29fd7540fa2cd3adcbc79ffcc81fb8d7244` | tiling 主逻辑；InitParams；CalcUbBmm |
| `op_host/sparse_flash_attention_tiling.h` | `fd1305d10eb477fad3d726961e96c4bb9420c5761f73c9d5bd6dd142ef40d6a8` | SparseFlashAttentionTilingDataMla L196-203 |
| `op_kernel/sparse_flash_attention.cpp` | `6a5dc0293c7118672b4686f1db966d892e0efb09443113d7afa6457570eb5858` | 模板入口 L32-54；dtype dispatch L46-53；MIX_AIC_1_2 L41 |
| `op_kernel/sparse_flash_attention_template_tiling_key.h` | `-` | 模板参数 L31-37；4 合法组合 L41-69；C/V_TEMPLATE L27-28 |
| `op_kernel/sparse_flash_attention_common.h` | `5549e6a85832132d7ecdcd7025afb0a521055cf3165567d4a1034a00e74504b8` | TEMPLATE_MODE L36；ConstInfo 同步事件 L137-142；BlockAlign L59-66 |
| `op_kernel/sparse_flash_attention_kernel_mla.h` | `70d97224fb74c31bd031f815facf3b80f9b725430d6c8da48ccc981a56d96956` | PreloadPipeline L852-894；ComputeMm1/Mm2 L729-754；Process L758-775；CalcSinnerTopKBegin L941-1016；同步常量 L84-93 |
| `op_kernel/sparse_flash_attention_service_cube_mla.h` | `d95b92a8bb76acbffba2a9c10d77bc13b140911b8d37126b894d8619ff29c43a` | DataCopyGmNDToL1 L47-64；DataCopyPA L72-113；InitBuffers L301-317 |
| `op_kernel/sparse_flash_attention_service_vector_mla.h` | `e3aad0cbed428dab2e8d11504060e3d78e6cab0cab27e82f39480abd7ebfae87` | MergeKv L986；GetRealS2Idx L821-831；GetKeyGmOffset L834-853；CopyInKv L896-957；CopyOutMrgeResult L960-982；InitBuffers L211-248 |

## 3. 关键代码区域摘要

### 3.1 计算公式（README L16-20）
`softmax(Q·K̃^T/√dk)·Ṽ`，K̃/Ṽ 为 sparse_indices 选取的稀疏重要 K/V。

### 3.2 算子定义（def.cpp L23-88）
- 输入：query/key/value（必选，FP16/BF16）、sparse_indices（必选，INT32）、block_table/actual_seq_lengths_query/_kv/query_rope/key_rope（可选）。
- 输出：attention_out（必选）、softmax_max/softmax_sum（return_softmax_lse=True 时）。
- 属性：scale_value(必选)、sparse_block_size(默认1)、layout_query/layout_kv(默认BSND)、sparse_mode(默认3)、attention_mode(仅2=MLA-absorb)、return_softmax_lse(默认false)。
- AICore config：ascend910b + ascend910_93（仅 A2/A3 推理）。

### 3.3 MLA-absorb（README L73）
attention_mode=2：Q/K 的 nope(512) 与 rope(64) 沿 D 维拼接成 576 用于 QK；key/value 共享同一底层张量。

### 3.4 模板分发（template_tiling_key.h L31-69）
模板参数 FLASH_DECODE/LAYOUT_T/KV_LAYOUT_T/TEMPLATE_MODE，4 组合法组合；dtype dispatch FP16/BF16（kernel 入口 L46-53）；任务类型 KERNEL_TYPE_MIX_AIC_1_2（AIC:AIV=1:2）。C/V_TEMPLATE 由 `sparseBlockSize≤4 && s2Size≠0` 触发。

### 3.5 流水与离散 KV 搬运
5 阶段 CV 流水（V_TEMPLATE 含 MergeKv 前置），3-slot PreloadPipeline，5 个 CV sync event；MergeKv 将离散 sparse block 聚合为连续 chunk。详见 [[technique-sparse-attention-pipeline]]、[[technique-sparse-kv-gather]]。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用：
- `wiki/kernels/sparse-flash-attention.md`
- `wiki/techniques/sparse-attention-tiling.md`
- `wiki/techniques/sparse-attention-pipeline.md`
- `wiki/techniques/sparse-kv-gather.md`
- `artifacts/sparse-flash-attention/`（source_file_inventory.md、tiling_dispatch_tables.md、migration-table.md、sparse-flash-attention-triton-skeleton.py）

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数数值、UB/L1 占用、FlashDecoding 合并数值稳定性、PA offset 算术等需结合硬件环境进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
