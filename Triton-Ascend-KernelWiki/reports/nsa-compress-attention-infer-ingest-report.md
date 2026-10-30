# nsa_compress_attention_infer CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `nsa_compress_attention_infer` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred（CANN 仓库 `attention/` 目录分类；tags.yaml#kernel_types 含 `attention`） |
| VARIANTS | `infer` | inferred（NSA 推理流水线第一阶段） |
| mode | static_ingest | - |

## 2. Source file inventory

算子目录 `attention/nsa_compress_attention_infer/`（commit `d28d8f9...`）。核心文件：

| 文件 | role | centrality | 关键区域 |
|---|---|---|---|
| `docs/aclnnNsaCompressAttentionInfer.md` | doc/api | core | NSA 公式 L22-28；参数表 L77-353；产品支持 L5-12 |
| `op_host/nsa_compress_attention_infer_def.cpp` | host/def | core | 输入 L19-56；输出 L57-63；属性 L64-86；AICore L88-89 |
| `op_host/nsa_compress_attention_infer_tiling.cpp` | host/tiling | core | Tiling L457-558；AssignTask L327-445 |
| `op_kernel/nsa_compress_attention_infer.h` | kernel | core | Aic L40-689；Aiv L690-1565；ProcessMm1 L517-559；ProcessTopK L1479-1526 |
| `op_kernel/ncai_public_define.h` | kernel/common | core | NCAIType L46-52；sync event id L34-36 |

完整清单见 `artifacts/nsa-compress-attention-infer/source_file_inventory.md`。

## 3. Operator semantics

- **NSA 公式**: `P_cmp=Softmax(scale·Q·K^T)` → `attentionOut=P_cmp·V` → `P_slc[j]=sum over sliding window` → `topKIndices=topk(P_slc')`。
- **输入**: query（必选，FP16/BF16，TND/BSND）、key/value（3D `[numBlocks,blockSize,N*D]` PagedAttention）、blockTable（必选）、actualCmpKvSeqLen（必选）、actualQSeqLen/actualSelKvSeqLen（可选）。
- **输出**: output（同 query）、topKOutput（INT32，topK 块索引）。
- **属性**: numHeads、numKeyValueHeads、selectBlockSize（16-128）、selectBlockCount、compressBlockSize（16-128）、compressBlockStride（16-64）、scaleValue、pageBlockSize（≤128）、sparseMode。
- **约束**: A2/A3 训练+推理 √；其余 ×。PagedAttention 必选。

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|---|---|---|---|
| QK^T | ProcessMm1（.h L517-559） | AIC | Q·K^T，blockTable 映射 KV |
| Softmax | ProcessSoftmax（.h L1083-1116） | AIV | 含 mask、scale |
| PV | ProcessMm2（.h L562-608） | AIC | 加权求和 |
| 重要性分数 | ProcessImportanceScore（.h L1258-1273） | AIV | 滑动窗口卷积 |
| topK | ProcessTopK（.h L1479-1526） | AIV | 选择最重要块索引 |

## 5. Tiling parameters

- 2 key（0=TND，1=BSND），不区分 dtype。
- 核间切分：B×N2×Q-S。
- blockSize≤128（16 倍数）、compressBlockSize/selectBlockSize 仅支持 16/32/48/64/80/96/112/128。
- workspace：mm1Res（FP32）、mm2In（FP16/BF16）、scoreIn（FP32）、topKIn（FP32），2× 双缓冲。

## 6. Core partition strategy

- 任务=（batch, kvHead, Q-token），`qSeqLenSplitSize * kvHeadNum`。
- Q_S≤4 多 token 路径，单 token 自动降级。

## 7. Kernel templates & dispatch

- 2 kernel template（FP16/BF16 编译期选择），2 tiling key（TND/BSND）。
- MIX_AIC_1_2（Cube:Vector=1:2），Cube/Vector 分离实现。
- 不使用 `attn_infra` 模板库（与 block_sparse_attention 系列不同），NSA 独立实现。

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|---|---|---|---|---|
| QK^T | ProcessMm1 | 注意力分数 | tl.dot | rewrite |
| softmax | ProcessSoftmax | 归一化 | tl.exp/max/sum | transferable |
| PV | ProcessMm2 | 加权求和 | tl.dot | rewrite |
| importance score | ProcessImportanceScore | 滑动窗口卷积 | 三角形权重累加 | transferable |
| topK | ProcessTopK | 选择索引 | tl.argsort | transferable |
| Paged KV | blockTable 映射 | 间接访问 | tl.load(blockTable) | rewrite |
| 同步 | CrossCore flag | Cube↔Vector 依赖 | 不迁移 | CANN-specific |
| 双缓冲 | ping-pong flag | 流水 | 不迁移 | CANN-specific |

## 9. CANN -> Triton-Ascend migration classification

完整表见 `artifacts/nsa-compress-attention-infer/migration-table.md`。

## 10. Triton-Ascend skeleton

参考 `artifacts/nsa-compress-attention-infer/nsa-compress-attention-infer-triton-skeleton.py`。reproducibility: pseudocode。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `sources/ops-transformer/nsa-compress-attention-infer.md` | source | ops-transformer-nsa-compress-attention-infer | - | - | - |
| `wiki/kernels/nsa-compress-attention-infer.md` | kernel | kernel-nsa-compress-attention-infer | ops-transformer-nsa-compress-attention-infer | inferred | pseudocode |
| `artifacts/nsa-compress-attention-infer/`（4 files + PROVENANCE） | artifact | - | - | - | - |

## 12. Validation & retrieval

- validate.py: OK（exit 0）
- generate-indices.py: 5 索引生成
- query.py nsa_compress_attention_infer --compact: 2 命中（source）
- query.py compress_attention --compact: 含 kernel-nsa-compress-attention-infer

## 13. Unverified / needs-hardware items

- 滑动窗口卷积三角形权重细节
- tiling 参数数值（qSeqLenSplitSize 等）
- UB/L1 字节预算 / workspace 占用

## 14. Modified / created files

创建：
- `kernelwiki/sources/ops-transformer/nsa-compress-attention-infer.md`
- `kernelwiki/wiki/kernels/nsa-compress-attention-infer.md`
- `kernelwiki/artifacts/nsa-compress-attention-infer/source_file_inventory.md`
- `kernelwiki/artifacts/nsa-compress-attention-infer/tiling_dispatch_tables.md`
- `kernelwiki/artifacts/nsa-compress-attention-infer/migration-table.md`
- `kernelwiki/artifacts/nsa-compress-attention-infer/nsa-compress-attention-infer-triton-skeleton.py`
- `kernelwiki/artifacts/nsa-compress-attention-infer/PROVENANCE.yaml`
- `reports/nsa-compress-attention-infer-ingest-report.md`