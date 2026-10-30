---
id: ops-transformer-lightning-indexer
title: CANN ops-transformer LightningIndexer source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/lightning_indexer/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-22
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer LightningIndexer 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/lightning_indexer/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。本算子为稀疏注意力（SparseFlashAttention）的索引选择器，根据 Q_index 和 K_index 计算重要性分数后 topK 选出每个 token 的最重要 KV 位置。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/lightning_indexer/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-22
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单见 `artifacts/lightning-indexer/source_file_inventory.md`。关键文件：

| 源文件 | SHA-256 | 关键区域 |
|---|---|---|
| `README.md` | `f19f9dbca76399be...` | 公式 L20-24；参数 L39-71；约束 L80-86；产品支持 L5-12 |
| `op_host/lightning_indexer_def.cpp` | `adeea6351c871607...` | 输入 L23-37；输出 L53-60；属性 L61-67；AICore L75-76 |
| `op_host/lightning_indexer_tiling.h` | `9ba89b537c8def48...` | `LITilingData` L79-93；`LITilingInfo` L121-154 |
| `op_kernel/lightning_indexer_kernel.h` | `8e757fd849a6bee3...` | `LIPreload` L53-153；`SplitCore` L279-357 |
| `op_kernel/lightning_indexer_service_cube.h` | `12b8bc0b93318b38...` | `LIMatmul` L28-80；`ComputeMm1` |
| `op_kernel/lightning_indexer_service_vector.h` | `938d11abc7002fbc...` | `LIVector` L50-112；`ProcessVec`/`ProcessLD` |
| `op_kernel/lightning_indexer_template_tiling_key.h` | `a087348920a255ae...` | 24 模板组合 L45-86 |

## 3. 关键代码区域摘要

### 3.1 公式（README L20-24）
`Indices = Top-k{[1]_{1×g} @ [(W @ [1]_{1×S_k}) ⊙ ReLU(Q_index @ K_index^T)]}`。简化为：计算 Q·K^T，ReLU 激活，weights 缩放，g 组求和，topK 选出最重要的 sparse_count 个位置。

### 3.2 算子定义（def.cpp L19-77）
输入：query/key（FP16/BF16，BSND/TND/PA_BSND）、weights（FP16/BF16/FP32）、actual_seq_lengths（可选）、block_table（可选，PA 场景）。输出：sparse_indices（INT32 `[B,S1,N2,sparseCount]`/`[T1,N2,sparseCount]`）、sparse_values（可选，同 key dtype）。属性：layout_query（BSND/TND）、layout_key（BSND/TND/PA_BSND）、sparse_count（2048）、sparse_mode（3 c rightDownCausal）、pre_tokens/next_tokens、return_values。AICore：ascend910b + ascend910_93。

### 3.3 kernel 流水
Cube（AIC）：`ComputeMm1`（Q·K^T，含 PageAttention 路径）→ workspace。Vector（AIV）：`ProcessVec`（ReLU + weights 缩放 + topK 选择）→ `ProcessLD`（sparse_count>2048 时分布式 merge）→ 输出 sparse_indices/values。24 模板组合（dtype×PA×layout×weights_flag）。

### 3.4 产品支持
A2（910B）√、A3（910_93）√；950PR/950DT ×。

## 4. 衍生引用

- `wiki/kernels/lightning-indexer.md`
- `artifacts/lightning-indexer/`

## 5. 复现性说明

本页仅记录源文件路径、SHA-256 与关键代码区域摘要。要复现分析需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。