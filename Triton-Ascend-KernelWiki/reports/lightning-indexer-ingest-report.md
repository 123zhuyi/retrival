# lightning_indexer CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `lightning_indexer` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred |
| VARIANTS | `forward`（稀疏注意力索引选择器） | inferred |
| mode | static_ingest | - |

## 2. Source file inventory

核心文件见 `artifacts/lightning-indexer/source_file_inventory.md`。

## 3. Operator semantics

- **公式**: `Indices = Top-k([1]_{1×g} @ [(W @ [1]_{1×S_k}) ⊙ ReLU(Q_index @ K_index^T)])`
- **输入**: query/key（FP16/BF16）、weights（FP16/BF16/FP32）、block_table（PA 场景）
- **输出**: sparse_indices（INT32）、sparse_values（可选）
- **属性**: layout_query（BSND/TND）、layout_key（BSND/TND/PA_BSND）、sparse_count、sparse_mode、return_values
- **约束**: N1≤64、N2=1、D=128、sparse_count∈[1,2048]∪{3072~8192}

## 4. Pipeline

Cube（AIC）：Q·K^T（含 PA blockTable）→ Vector（AIV）：ReLU + weights 缩放 + topK 选择 + 可选 LD merge（sparse_count>2048 时）→ 输出 sparse_indices/values。24 模板组合。

## 5. Validation

- validate.py: **OK: KernelWiki validation passed**
- generate-indices.py: 5 索引生成
- query.py lightning_indexer --compact: 6 命中

## 6. Pages written

| Path | type | id | sources |
|---|---|---|---|
| `sources/ops-transformer/lightning-indexer.md` | source | ops-transformer-lightning-indexer | - |
| `wiki/kernels/lightning-indexer.md` | kernel | kernel-lightning-indexer | ops-transformer-lightning-indexer |
| `artifacts/lightning-indexer/`（4 files + PROVENANCE） | artifact | - | - |