# moe_init_routing CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `moe_init_routing` | source-confirmed |
| OPERATOR_TYPE | `moe` | inferred |
| VARIANTS | `forward` | inferred |
| mode | static_ingest | - |

## 2. Source file inventory

核心文件见 `artifacts/moe-init-routing/source_file_inventory.md`。

## 3. Operator semantics

- **公式**: `sort(expertIdx, rowIdx)` → `expandedRowIdx[sortedRowIdx[i]]=i` → `expandedX[i]=x[sortedRowIdx[i]%numRows]`
- **输入**: x（FP16/BF16/FP32）、rowIdx/expertIdx（INT32）
- **输出**: expandedX、expandedRowIdx、expandedExpertIdx
- **属性**: activeNum

## 4. Pipeline

三段流水（纯 Vector，MIX_AIV_1_0）：sort(expertIdx, rowIdx) → srcToDst（映射）→ gather（expandedX）。5 tiling key。

## 5. Validation

- validate.py: **OK**
- generate-indices.py: 5 索引
- query.py "moe" --compact: 5 命中（含 kernel-moe-init-routing）

## 6. Pages written

| Path | type | id |
|---|---|---|
| `sources/ops-transformer/moe-init-routing.md` | source | ops-transformer-moe-init-routing |
| `wiki/kernels/moe-init-routing.md` | kernel | kernel-moe-init-routing |
| `artifacts/moe-init-routing/`（4 files + PROVENANCE） | artifact | - |