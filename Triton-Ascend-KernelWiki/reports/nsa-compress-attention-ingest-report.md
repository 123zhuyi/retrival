# nsa_compress_attention CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer/attention/nsa_compress_attention/` | source-confirmed |
| KERNELWIKI_ROOT | `/home/w00836812/shixi_test02/Triton-Ascend-KernelWiki/kernelwiki` | source-confirmed |
| OPERATOR_NAME | `nsa_compress_attention` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred (CANN ops-transformer/attention 目录，tags.yaml#kernel_types 含 attention) |
| VARIANTS | `forward` | inferred (前向压缩注意力，无反向) |
| mode | `static_ingest` | - |

## 2. Source file inventory

完整表见 `artifacts/nsa-compress-attention/source-file-inventory.md`（~25 文件，按 op_host/op_kernel/op_api/docs/tests 分角色与中心度）。

核心文件（相对算子目录）：

| path | role | centrality | key symbols |
|------|------|-----------|-------------|
| `op_host/nsa_compress_attention_def.cpp` | 定义 | core | OpDef I/O、属性(compressBlockSize/compressStride/selectBlockSize/selectBlockCount)、AICore 配置 ascend910b/910_93 |
| `op_host/nsa_compress_attention_tiling_general.cpp` | tiling | core | NsaCompressAttentionTilingBase + NsaCompressAttentionVarLenScoreTiling |
| `op_kernel/nsa_compress_attention.cpp` | kernel入口 | core | 8 分支 dispatch（FP16/BF16 × hasAttenMask × hasTopkMask） |
| `op_kernel/nsa_compress_attention_s1s2_bn2gs1_sab.h` | kernel模板 | core | NsaCompressAttentionS1s2Bn2gs1SameAB |

## 3. Operator semantics

- **输入**：query/key/value（FP16/BF16）、atten_mask（BOOL/UINT8）、topk_mask、actual_seq_qlen/cmp_seq_kvlen/sel_seq_kvlen（INT64 前缀和）
- **输出**：softmax_max/softmax_sum（FLOAT）、attention_out（FP16/BF16）、topk_indices_out（INT32）
- **属性**：scale_value(1.0)、head_num(必需)、input_layout(必需，仅TND)、sparse_mode(0)、compress_block_size(l, 必需)、compress_stride(d, 必需)、select_block_size(l', 必需)、select_block_count(n, 必需)
- **核心公式（6 步）**：`P_cmp=Softmax(QK^T)`；`attention_out=Softmax(mask(scale·QK^T))·V`；`P_slc[j]=∑_{m=0}^{l'/d-1}∑_{n=0}^{l/d-1}P_cmp[l'/d·j−m−n]`；`P_slc'=∑_h P_slc^h`；`P_slc'=topk_mask(P_slc')`；`topkIndices=topk(P_slc')`
- **约束**：l 16~128，d 16~64，l' 16~128，l'%d==0，l'≥l≥d；n[1,32]≤min(SelSkv)；CmpSkv≤14000；G<128&&128%G==0；仅TND；仅ascend910b/910_93
- **动态 shape**：TND varlen（actual_seq_qlen/cmp_seq_kvlen/sel_seq_kvlen 前缀和）

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|-------|-------|------|----|
| Cube1 | IterateBmm1 | AIC | Q·K^T matmul |
| Vector1 | ProcessVec1 | AIV | scale → mask → Softmax → 写回 workspace |
| Cube2 | IterateBmm2 | AIC | P·V matmul → attention_out |
| Vector2 | ProcessVec2 + TopkCompute | AIV | importance score 滑动窗口 + 跨 head 累加 + topk_mask + topk 索引 |

## 5. Tiling parameters

- `s1BasicBlock = 128 / gSize`（NSA_BASE_S1G_SIZE / gSize）
- `s1VecBasicBlock = s1BasicBlock / 2`（CV_RATIO=2）
- `s2BasicBlock = alignedS2`（S2 不做切分，整块计算）
- 核间切分：`totalSize = accumS1BlockNum * n2Size`，`coreNum = min(totalSize, aicNum) * 2`（SameAB AIC×2）
- workspace：`pWorkspaceSize + impScoreWorkspaceSize + stage1ResWorkspaceSize + 16MB`
- UB：`availUbSize = 192KB`；Softmax 区域 + importance score 区域 + topk 区域

## 6. Core partition strategy

- `accumS1BlockNum = Σ CeilDiv(actualSeqLenData[i], s1BasicBlock)` 跨 batch 累加
- `actualUsedAivNum = min(totalSize, aivNum)`；`actualUsedAicNum = min(totalSize, aicNum)`
- `blockDim = CalcTschBlockDim(coreNum, aicNum, aivNum)`（PostTiling）
- S2 不做切分（整块计算）

## 7. Kernel templates & dispatch

- **1 个 kernel 模板类**：`NsaCompressAttentionS1s2Bn2gs1SameAB`（`nsa_compress_attention_s1s2_bn2gs1_sab.h`）
- 1 个 varlen 变体：`NsaCompressAttentionVarLenScoreSameAB`（`nsa_compress_attention_var_len_score_sab.h`）
- 8 分支 dispatch：tiling key 固定 `10000000000000000024UL`，按 dtype（FP16/BF16）× hasAttenMask × hasTopkMask 区分
- 可迁移触发：dtype / hasAttenMask / hasTopkMask → Triton constexpr 分支
- 不迁移：tiling key bit 编码、REGISTER_OPS_TILING_TEMPLATE、核型绑定

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|-------|---------------|---------|-------------------|---------------|
| QK^T | Cube IterateBmm1，L0C FP32 累加 | 步 1-2 | `tl.dot(q,k.T)` FP32 | 是 |
| mask + scale | Vector SelectWithBytesMask + Muls | 步 2 | `tl.where` + *sm_scale | 是 |
| Softmax | Vector SoftMax API | 步 2 | 在线 softmax | 是 |
| P·V | Cube IterateBmm2 | 步 2 | `tl.dot(p,v)` | 是 |
| importance score 滑动窗口 | DataCopyPad + ConfusionTranspose + Muls + Add | 步 3 | 滑动窗口循环 | 是 |
| 跨 head 累加 | Add 循环 reduce g | 步 4 | `tl.sum` 跨 G 轴 | 是 |
| topk | TopK API | 步 6 | `tl.topk` | 是 |
| 写回 | DataCopy + DataCopyPad | 输出 | `tl.store` masked | 是 |
| 双缓冲 | mm1Res[0]/[1] double buffer | 搬运/计算重叠 | 编译器排流 | 意图迁移 |
| CV 同步 | SetFlag/WaitFlag + PipeBarrier | CV 依赖 | 不移植 | 否（CANN-specific） |
| ConfusionTranspose | 手动 ND→ND transpose | 数据重排 | 不移植 | 否（CANN-specific） |

## 9. CANN -> Triton-Ascend migration classification

完整表见 `artifacts/nsa-compress-attention/migration-table.md`。摘要：
- **transferable**：NSA 公式、滑动窗口、跨 head 累加、topk、mask、FP32 累加、尾块、约束
- **rewrite**：单模板、CV tiling 分离、核间切分、importance score UB 流水、workspace 分配
- **CANN-specific（不迁移）**：MTE/DataCopy、BufferID/queue/event、手动 CV 时序、ConfusionTranspose、TBuf 放置
- **needs_verification**：基本块数值最优性、UB 占用、CmpSkv≤14000 限制、TND 布局偏移、性能

## 10. Triton-Ascend skeleton

参考 `artifacts/nsa-compress-attention/nsa-compress-attention-triton-skeleton.py`。
`reproducibility: pseudocode`。未编译、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|------|------|----|---------|-----------|----------------|
| `kernelwiki/sources/ops-transformer/nsa-compress-attention.md` | source | ops-transformer-nsa-compress-attention | - | - | - |
| `kernelwiki/wiki/kernels/nsa-compress-attention.md` | kernel | kernel-nsa-compress-attention | [ops-transformer-nsa-compress-attention] | inferred | pseudocode |
| `kernelwiki/artifacts/nsa-compress-attention/PROVENANCE.yaml` | provenance | - | - | - | - |
| `kernelwiki/artifacts/nsa-compress-attention/source-file-inventory.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/nsa-compress-attention/tiling-dispatch-tables.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/nsa-compress-attention/pipeline-diagram.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/nsa-compress-attention/migration-table.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/nsa-compress-attention/nsa-compress-attention-triton-skeleton.py` | derived-source | - | - | - | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**
- `generate-indices.py`: 生成 5 个索引
- `query.py nsa_compress_attention --compact`: 预期 2 命中（source 页 + kernel 页）
- `query.py attention tiling --compact`: 预期 10+ 命中（含 nsa-compress-attention）
- `query.py attention mask --compact`: 预期 10+ 命中
- `query.py attention triton ascend --compact`: 预期 10 命中（triton-ascend 文档源页）

## 13. Unverified / needs-hardware items

- tiling 基本块数值（NSA_BASE_S1G_SIZE=128 等）是否最优 -- 需硬件实测
- UB 占用（availUbSize=192KB vs 实际使用）-- 需 profile
- `CmpSkv ≤ 14000` 的 UB 限制 -- 需硬件确认
- TND 布局偏移算术 -- 需张量实例确认
- topk UB 参数（topkBase 计算）-- 需硬件确认
- 性能（任何加速比）-- `static_ingest` 无 baseline

## 14. Modified / created files

- `kernelwiki/artifacts/nsa-compress-attention/PROVENANCE.yaml` — bundle 溯源
- `kernelwiki/artifacts/nsa-compress-attention/source-file-inventory.md` — S1 源文件清单
- `kernelwiki/artifacts/nsa-compress-attention/tiling-dispatch-tables.md` — S4/S5 tiling + dispatch 表
- `kernelwiki/artifacts/nsa-compress-attention/pipeline-diagram.md` — S6 流水抽象
- `kernelwiki/artifacts/nsa-compress-attention/migration-table.md` — S3 迁移分类
- `kernelwiki/artifacts/nsa-compress-attention/nsa-compress-attention-triton-skeleton.py` — S7 Triton skeleton
- `kernelwiki/sources/ops-transformer/nsa-compress-attention.md` — source 页
- `kernelwiki/wiki/kernels/nsa-compress-attention.md` — kernel wiki 页
- `reports/nsa-compress-attention-ingest-report.md` — 本报告
- `kernelwiki/queries/*.md` — 由 generate-indices.py 重新生成

## Quality checklist

- [x] Every `sources:` resolves to a source page id；每个 wiki 页 cite source
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 为 tags.yaml 精确值
- [x] `validate.py` 通过
- [x] `performance_claims: []`（无 baseline）
- [x] `confidence: inferred`；`reproducibility: pseudocode`
- [x] 无 "runs/verified/benchmarked" 主张
- [x] CANN 机制（MTE/BufferID/queue/event/手动同步/ND↔NZ）未作为 Triton 指导
- [x] 报告列出所有创建/编辑文件