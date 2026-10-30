# nsa_selected_attention CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `nsa_selected_attention` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred（CANN 仓库 `attention/` 目录分类；tags.yaml#kernel_types 含 `attention`） |
| VARIANTS | `forward`（训练前向，输出 softmaxMax/Sum 供反向用） | inferred |
| mode | static_ingest | - |

## 2. Source file inventory

算子目录 `attention/nsa_selected_attention/`（commit `d28d8f9...`）。核心文件：

| 文件 | role | centrality | 关键区域 |
|---|---|---|---|
| `docs/aclnnNsaSelectedAttention.md` | doc/api | core | 公式 L22-28；参数表 L68-353；约束 L358-380 |
| `op_host/nsa_selected_attention_tiling_general.cpp` | host/tiling | core | GetTilingKey 0/1 L565-572；CalcWorkspaceSize L637-650 |
| `op_kernel/nsa_selected_attention.h` | kernel | core | Process() L359-392；SelectAndGatherK L447-481；ProcessSoftmax L524-581 |
| `op_kernel/nsa_selected_attention_common.h` | kernel/common | core | SOFTMAX_REDUCE_SIZE=8 L56 |

完整清单见 `artifacts/nsa-selected-attention/source_file_inventory.md`。

## 3. Operator semantics

- **公式**: `Gather(K, topKIndices)` → `Softmax(scale·Q·selected_K^T)·selected_V`
- **输入**: query/key/value（FP16/BF16，仅 TND）、topkIndices（INT32，必选）、attenMask（可选）
- **输出**: softmaxMax/Sum（FLOAT `[T,N1,8]`）、attentionOut（同 query）
- **属性**: scaleValue、headNum、sparseMode（0/2）、selectedBlockSize（≤128 16倍数）、selectedBlockCount（[1,128]）
- **约束**: 仅 A2 训练（A3 ×）、仅 TND、D_qk=192 D_v=128、GQA gSize≤32

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|---|---|---|---|
| Gather K | SelectAndGatherK（.h L447-481） | AIC | 按 topKIndices gather K 块 |
| QK^T | IterateBmm1（.h L688-715） | AIC | Q·selected_K^T |
| Gather V | SelectAndGatherV（.h L484-521） | AIC | 与 BMM1 重叠执行 |
| 在线 softmax | ProcessSoftmax（.h L524-581） | AIC | 含 causal mask，输出 max/sum |
| PV | IterateBmm2（.h L725-739） | AIC | P·selected_V |

## 5. Tiling parameters

- 2 key（0=无mask，1=causal）。空输入另 3 key（90/92/94）。
- 核间切分：S1（query 行）→ `coreNum = min(aivNum, t1Size)`。
- GQA：`gSize = N1/N2`，Q 布局 `[S1,N2,G,D]`。
- workspace：per-core selectedK/V + P + softmax 残差。

## 6. Core partition strategy

- S1 按行切分；`headCoreNum` 处理 `ceil(t1Size/coreNum)` 行，`tailCoreNum` 处理少一行。
- 每核内循环 N2。

## 7. Kernel templates & dispatch

- `NsaSelectedAttention<INPUT_T, ATTEN_ENABLE>`：2 dtype（FP16/BF16 编译期）× 2 mask（0/1）= 4 特化。
- Cube/Vector 路径（C220 MIX_AIC_1_2）。
- 不使用 attn_infra 模板库，NSA 独立实现。

## 8. Pipeline analysis

| Stage | CANN | Triton | 可迁移 |
|---|---|---|---|
| topKIndices gather | SelectAndGatherK/V | `tl.load(idx) -> tl.load(k+idx*...)` | rewrite |
| BMM1/BMM2 | IterateBmm1/2 | `tl.dot` | rewrite |
| 在线 softmax | ProcessSoftmax | `tl.max/sum + exp` | transferable |
| softmaxMax/Sum 输出 | DataCopy maxUb/sumUb | `tl.store` | transferable |
| HandleMask（causal） | HandleMask | `tl.where` | rewrite |
| V gather 与 BMM1 重叠 | 流水 | 不迁移 | CANN-specific |

## 9. CANN -> Triton-Ascend migration classification

完整表见 `artifacts/nsa-selected-attention/migration-table.md`。

## 10. Triton-Ascend skeleton

参考 `artifacts/nsa-selected-attention/nsa-selected-attention-triton-skeleton.py`。reproducibility: pseudocode。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/nsa-selected-attention.md` | source | ops-transformer-nsa-selected-attention | - | - | - |
| `wiki/kernels/nsa-selected-attention.md` | kernel | kernel-nsa-selected-attention | ops-transformer-nsa-selected-attention | inferred | pseudocode |
| `artifacts/nsa-selected-attention/`（4 files + PROVENANCE） | artifact | - | - | - | - |

## 12. Validation & retrieval

- validate.py: OK（exit 0）
- generate-indices.py: 5 索引生成
- query.py nsa_selected_attention --compact: 4 命中（source + NSA 家族）
- query.py "selected attention" --compact: 含 kernel-nsa-selected-attention

## 13. Unverified / needs-hardware items

- D_qk=192, D_v=128 固定值
- 仅 A2 训练（A3 不支持）
- 仅 TND（无 BSND 路径）
- softmaxMax/Sum shape `[T,N1,8]` dim=8 含义
- tiling 参数数值

## 14. Modified / created files

创建：
- `kernelwiki/sources/ops-transformer/nsa-selected-attention.md`
- `kernelwiki/wiki/kernels/nsa-selected-attention.md`
- `kernelwiki/artifacts/nsa-selected-attention/source_file_inventory.md`
- `kernelwiki/artifacts/nsa-selected-attention/tiling_dispatch_tables.md`
- `kernelwiki/artifacts/nsa-selected-attention/migration-table.md`
- `kernelwiki/artifacts/nsa-selected-attention/nsa-selected-attention-triton-skeleton.py`
- `kernelwiki/artifacts/nsa-selected-attention/PROVENANCE.yaml`
- `reports/nsa-selected-attention-ingest-report.md`