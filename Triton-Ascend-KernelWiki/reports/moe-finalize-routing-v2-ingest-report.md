# moe_finalize_routing_v2 CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `.../Triton-Ascend-KernelWiki` | source-confirmed |
| OPERATOR_NAME | `moe_finalize_routing_v2` | source-confirmed |
| OPERATOR_TYPE | `moe` | source-confirmed |
| VARIANTS | `forward,inference,training` | inferred |
| mode | `static_ingest` | - |

## 2. Source file inventory

算子目录 `moe/moe_finalize_routing_v2/`，源码 ~16k 行（含 test）。MoE FFN 输出合并算子（V2），三架构（A2/A3 + 950 arch35）+ 310P + Kirin。核心是 gather + scale + bias 加权求和。含 arch35 regbase 模板。

完整清单见 `artifacts/moe-finalize-routing-v2/source-file-inventory.md`（注：文件名用连字符）。核心文件：`README.md`、`def.cpp`（AICore config 910b/910_93/950/310p/Kirin）、`moe_finalize_routing_v2.cpp`（入口）、14+ 个 bf16/fp 模板、arch35 regbase 模板。

## 3. Operator semantics

- 输入（7）：expandedX/expandedRowIdx/x1Optional/x2Optional/biasOptional/scalesOptional/expertIdxOptional。
- 输出（1）：out。
- 公式：`out(i,j) = x1 + x2 + Σ_k (scales(i,k) * (expandedX[expandedRowIdx(i+k*R), j] + bias[expertIdx[i,k], j]))`。
- 属性：dropPadMode（排列方式）。
- V2 相对 V1：多 950（arch35 regbase）+ 310P + Kirin；输入改 Optional；新增 dropPadMode。
- 类型：`moe`。

## 4. Stage / pipeline table

| Stage | Unit | Op |
|---|---|---|
| expandedX gather | Vector | expandedRowIdx -> 间接 load expandedX |
| bias gather | Vector | expertIdx -> 间接 load bias |
| scale + add | Vector | scales * (expandedX + bias) 累加 |
| residual | Vector | + x1 + x2 |
| writeback | Vector | -> out |

纯 AIV-only Vector 算子（无 matmul）。

## 5. Tiling parameters

- ~50 个 tiling key，按 dtype × 算法策略（CutK/Db/CuthK2/CuthK4/Cuth/AllBias）× 是否有 bias。
- arch35 用 regbase 模板（h_full_load/h_partial_load/k_h_full_load/row_k_h_full_load）。
- 详见 `tiling-dispatch-tables.md`。

## 6. Core partition strategy

- 核间：沿 R/H/K 轴切分（按模板策略）。
- 核内：gather + scale + bias + reduce sum。

## 7. Kernel templates & dispatch

- 14+ 个模板（bf16/fp × cuth/cutk/cuth_k2/cuth_k4/db/all_bias）。
- arch35 regbase 模板（4 个：h_full_load/h_partial_load/k_h_full_load/row_k_h_full_load）。
- 架构：910b/910_93（arch32）/ 950（arch35）/ 310p / Kirin。

## 8. Pipeline analysis

| Stage | CANN mechanism | Triton abstraction | Transferable? |
|---|---|---|---|
| expandedX gather | expandedRowIdx + DataCopy | `tl.load(expandedX + idx)` | 是（语义） |
| bias gather | expertIdx + DataCopy | `tl.load(bias + expert_id * H)` | 是（语义） |
| scales 加权 | Mul + Add 累加 | `tl.sum(scales * val, axis=K)` | 是（数学） |
| residual | Add | `x1 + x2` | 是（数学） |
| MTE 搬运 | DataCopyPad | `tl.load`/`tl.store` | 否（机制） |

## 9. CANN -> Triton-Ascend migration classification

- **transferable**：gather 公式、scale 加权求和、bias 查表、residual。
- **rewrite**：模板 dispatch、cutk/cuth 分核、arch35 regbase。
- **CANN-specific**：MTE DataCopyPad、KERNEL_TYPE_AIV_ONLY。
- **needs verification**：tiling 数值、UB 预算。

## 10. Triton-Ascend skeleton

`artifacts/moe-finalize-routing-v2/moe-finalize-routing-v2-triton-skeleton.py`，`reproducibility: pseudocode`。体现 gather + scale + bias + reduce。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/moe-finalize-routing-v2.md` | source | `ops-transformer-moe-finalize-routing-v2` | - | - | - |
| `wiki/kernels/moe-finalize-routing-v2.md` | kernel | `kernel-moe-finalize-routing-v2` | `[ops-transformer-moe-finalize-routing-v2]` | inferred | pseudocode |
| `artifacts/moe-finalize-routing-v2/PROVENANCE.yaml` | provenance | - | - | - | - |
| 5 个 artifacts 文件 | approach-notes | - | - | inferred | - |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数数值 | 依赖运行时 shape/平台 |
| UB 字节预算 | 需硬件实测 |
| ~50 tiling key 精确映射 | 以源码为准 |

## 14. Modified / created files

| File | 用途 |
|---|---|
| `kernelwiki/artifacts/moe-finalize-routing-v2/PROVENANCE.yaml` | bundle 溯源（修正 upstream_repo + origin_url） |
| `kernelwiki/artifacts/moe-finalize-routing-v2/source-file-inventory.md` | S1（已有，Jul 31） |
| `kernelwiki/artifacts/moe-finalize-routing-v2/migration-table.md` | S3（已有） |
| `kernelwiki/artifacts/moe-finalize-routing-v2/tiling-dispatch-tables.md` | S4/S5（已有） |
| `kernelwiki/artifacts/moe-finalize-routing-v2/pipeline-diagram.md` | S6（已有） |
| `kernelwiki/artifacts/moe-finalize-routing-v2/moe-finalize-routing-v2-triton-skeleton.py` | S7（已有） |
| `kernelwiki/sources/ops-transformer/moe-finalize-routing-v2.md` | S8 source（已有） |
| `kernelwiki/wiki/kernels/moe-finalize-routing-v2.md` | S8 kernel（已有） |
| `reports/moe-finalize-routing-v2-ingest-report.md` | S10（本次补写） |