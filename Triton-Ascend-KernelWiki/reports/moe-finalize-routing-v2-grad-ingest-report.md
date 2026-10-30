# moe_finalize_routing_v2_grad CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki` | source-confirmed |
| OPERATOR_NAME | `moe_finalize_routing_v2_grad` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| VARIANTS | `backward,training` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `moe/moe_finalize_routing_v2_grad/`，源码 ~6k 行。MoE 路由最终化反向，scatter-add + 可选 scale 梯度。三架构（A2/A3/950），纯 Vector 计算，4 种 TilingKey。

完整清单见 `artifacts/moe-finalize-routing-v2-grad/source_file_inventory.md`。

## 3. Operator semantics

- 输入（6）：gradY/expandedRowIdx/expandedX/scales/expertIdx/bias。
- 输出（2）：gradExpandedXOut/gradScalesOut。
- 公式：3 种场景（无 scale / 有 scale / 有 scale+bias），scatter-add + reduce。
- 类型：`moe`（非 attention）。

## 4. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/moe-finalize-routing-v2-grad.md` | source | `ops-transformer-moe-finalize-routing-v2-grad` | - | - | - |
| `wiki/kernels/moe-finalize-routing-v2-grad.md` | kernel | `kernel-moe-finalize-routing-v2-grad` | `[ops-transformer-moe-finalize-routing-v2-grad]` | inferred | pseudocode |
| 6 个 artifacts 文件 | approach-notes / provenance | - | - | inferred | - |

## 5. Validation

- `validate.py`: **OK: KernelWiki validation passed**。

## 6. Modified / created files

| File | 用途 |
|---|---|
| `artifacts/moe-finalize-routing-v2-grad/PROVENANCE.yaml` | bundle 溯源 |
| 5 个 artifacts 分析文件 | S1/S3/S4/S5/S6/S7 |
| `sources/ops-transformer/moe-finalize-routing-v2-grad.md` | S8 source |
| `wiki/kernels/moe-finalize-routing-v2-grad.md` | S8 kernel |
| `reports/moe-finalize-routing-v2-grad-ingest-report.md` | S10 |