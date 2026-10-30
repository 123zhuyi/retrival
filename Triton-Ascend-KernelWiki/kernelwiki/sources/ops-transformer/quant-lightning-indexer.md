---
id: ops-transformer-quant-lightning-indexer
title: CANN ops-transformer QuantLightningIndexer source snapshot
type: source
repo: cann/ops-transformer
path: attention/quant_lightning_indexer/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer QuantLightningIndexer 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/quant_lightning_indexer/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。本仓库为本地制品目录，`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。

`QuantLightningIndexer` 是推理场景下 SparseFlashAttention（SFA）的前处理算子，选出关键的稀疏 token 并输出 sparse_indices 供 SFA 使用。支持 int8（arch32）和 fp8（arch35）量化输入。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/quant_lightning_indexer/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 远端: `https://gitcode.com/cann/ops-transformer.git`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|------|------|------|
| `attention/quant_lightning_indexer/README.md` | `ad3f78edb352fd61936b97c076c6320007d0fc46fec126d46193cd1462040f93` | 算子总览、公式、I/O参数、约束、调用示例 |
| `attention/quant_lightning_indexer/op_host/quant_lightning_indexer_def.cpp` | `e291aa09790862be560068afdd411ceeb2d740b59bd69d280da225e9d43bcf1b` | OpDef：I/O dtype/layout（int8/fp8）、属性（sparse_count/sparse_mode/layout_query/layout_key）、AICore配置 |
| `attention/quant_lightning_indexer/op_host/quant_lightning_indexer_tiling.cpp` | `880f9d6afbcfd84e49c3966039d603bdd07c729594564b9c92454a1f74c3bd8b` | tiling入口：QLITilingData、CalcS1S2、CalcCoreNum、PageAttention处理 |
| `attention/quant_lightning_indexer/op_host/quant_lightning_indexer_tiling.h` | `7cb3c4f953d4b0cb31a93ce299884bdb598da9af8df2ad625a0c71ab59ed50a6` | TilingData结构体、QLITilingInfo、QLIInfoParser |
| `attention/quant_lightning_indexer/op_kernel/quant_lightning_indexer.cpp` | `2c545d3b5af3ba03ded1ea291ed794599566d747d025d3007b0bbcf49a30ed69` | kernel入口：模板参数 DT_Q/DT_K/DT_OUT/PAGE_ATTENTION/Q_LAYOUT_T/K_LAYOUT_T |
| `attention/quant_lightning_indexer/op_kernel/quant_lightning_indexer_kernel.h` | `60ed5297f5a87267691f95526d8cfc146ffceae30295ae3c9c972353b5a0cf88` | QLIPreload kernel模板（QKCompute + IndexCompute + TopK + WriteBack）|
| `attention/quant_lightning_indexer/op_kernel/quant_lightning_indexer_template_tiling_key.h` | `58896b240220147db7fa880e277ea23180d1763a2809a83f3a336e0c1af126d8` | 6模板参数声明与合法组合（arch32 2组合、arch35 6组合）|
| `attention/quant_lightning_indexer/op_kernel/quant_lightning_indexer_common.h` | `34b7b68a4f3aed35cdf41237d51619d44af4d30f363e833206a288c26c6e17a1` | QLIType模板结构体、LI_LAYOUT宏 |

## 3. 关键代码区域摘要

### 3.1 核心公式（来源: README.md L16-20）

```
out = Top_k{ W @ [ ReLU((Scale_Q ⊙ Scale_K^T) ⊙ (Q^Quant @ K^Quant^T)) ] }
```

1. `Q^Quant @ K^Quant^T` — 量化 QK 相关性（int8/fp8 → FP32 累加）
2. `Scale_Q ⊙ Scale_K^T` 逐元素乘反量化系数
3. ReLU 过滤负相关信号
4. `W @` 加权聚合（weights 系数）
5. Top-k 索引选择

### 3.2 输入输出（来源: def.cpp L20-136）

- **输入**：query（int8/fp8）、key（int8/fp8）、weights（FP16/BF16）、query_dequant_scale（FP16/FLOAT）、key_dequant_scale（FP16/FLOAT）、actual_seq_lengths_query/key（INT32）、block_table（INT32）
- **输出**：sparse_indices（INT32）
- **属性**：query_quant_mode(0)、key_quant_mode(0)、layout_query(BSND)、layout_key(BSND)、sparse_count(2048)、sparse_mode(3)、pre_tokens(INT64_MAX)、next_tokens(INT64_MAX)

### 3.3 架构分发（来源: def.cpp L71-136）

- **ascend910b/ascend910_93（arch32）**：query/key=int8, weights=FP16, dequant_scale=FP16（L71-79）
- **ascend950（arch35）**：query/key=FP8_E4M3FN/HIFLOAT8, weights=BF16, dequant_scale=FLOAT（L81-135）

### 3.4 TilingKey 与模板 dispatch（来源: template_tiling_key.h）

6 模板参数：DT_Q/DT_K/DT_OUT/PAGE_ATTENTION/Q_LAYOUT_T/K_LAYOUT_T。arch32 2 组合（int8 × PA_BSND/BSND/TND），arch35 6 组合（fp8_e4m3fn/hifloat8 × PA_BSND/BSND/TND）。

### 3.5 TilingData（来源: tiling.h L84-95）

`QLITilingData`：bSize, n2Size, gSize, s1Size, s2Size, sparseCount, usedCoreNum, blockSize, maxBlockNumPerBatch, sparseMode。HEAD_DIM_LIMIT=128 固定 D=128。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/quant-lightning-indexer.md`
- `artifacts/quant-lightning-indexer/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数具体数值、UB 占用、PageAttention 运行时契约等需结合硬件环境进一步验证。本页为 `static_ingest` 静态分析，未运行/未验证/未 benchmark 算子。