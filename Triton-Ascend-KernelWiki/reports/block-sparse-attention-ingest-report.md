# block_sparse_attention CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/`（repo root: `.../Triton-Ascend-KernelWiki`） | source-confirmed |
| OPERATOR_NAME | `block_sparse_attention`（前向） | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred（CANN 仓库 `attention/` 目录分类；def.cpp 注册为 attention 算子；tags.yaml#kernel_types 含 `attention`） |
| VARIANTS | `forward`（反向 `block_sparse_attention_grad` 为独立算子，按家族惯例单独入库，本次不覆盖） | inferred |
| mode | static_ingest | - |

> 内部代号 **RFA = Rain Fusion Attention**（旧名）；`aclnn_rain_fusion_attention.h` 为公开 `aclnnBlockSparseAttention*` API 的别名头；早期别名 `FASparse`（Flash-Attention Sparse，`kernel_common.hpp` L175-176，标注逐步废弃）。

## 2. Source file inventory

算子目录 `attention/block_sparse_attention/`（commit `d28d8f9...`）。核心文件：

| 文件（相对算子目录） | role | centrality | 关键符号 / 行区域 |
|---|---|---|---|
| `docs/aclnnBlockSparseAttention.md` | doc/api | core | 公式 L18-22；参数 L77-401；约束 L487-509；产品支持 L5-12 |
| `op_host/block_sparse_attention_def.cpp` | host/def | core | 输入 L19-54；输出 L55-62；属性 L64-73；AICore(910b/910_93) L75-76 |
| `op_host/block_sparse_attention_tiling.cpp` | host/tiling | core | GenerateTilingKey L683-748；CalculateTaskSplit L557-603；CalculateWorkSpace L605-630；CheckSparsePattern L390-437 |
| `op_kernel/block_sparse_attention_kernel.h` | kernel | core | **Mask2IdxAndCount L93-248**；主循环 L447-731；cross-core flag L782-784 |
| `op_kernel/block_sparse_attention_tilingkey.h` | kernel/dispatch | core | RFA_BASE_TILING L28；12 key 常量 L31-65 |
| `op_kernel/block_sparse_attention_interface.cpp` | kernel | core | tile shape/分派策略 L68-90；模板实例化 L99-110 |
| `op_kernel/attn_infra/gemm/block/block_mmad_qk.hpp` | template/gemm | core | BlockMmadQK L152-227：沿 selectIdx 稀疏 gather K |
| `op_kernel/attn_infra/gemm/block/block_mmad_pv.hpp` | template/gemm | core | BlockMmadPV L121-186：沿 selectIdx 稀疏 gather V |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax[_low_prec].hpp` | template/epilogue | core | 在线 softmax fp32/fp16 |
| `op_kernel/kernel_common.hpp` | kernel/common | core | 同步 id L34-37；块常量 L39-84 |

完整清单（含行数、SHA-256、attn_infra 全部头）见 `artifacts/block-sparse-attention/source_file_inventory.md`。`attn_infra/` 为类 CUTLASS 模板库（arch/gemm/epilogue/layout/coord），与反向算子共享。

## 3. Operator semantics

- **公式**（docs L18-22）：`attentionOut = Softmax(scale · query · key_sparse^T + atten_mask) · value_sparse`，稀疏块 `blockShapeX×blockShapeY`，`blockSparseMask` 指定每个 Q 块选择的 KV 块。
- **输入**：query/key/value（必选，FP16/BF16，TND `[T,N,D]`/BNSD `[B,N,S,D]`，D∈{64,128}）；blockSparseMask（必选，INT8 `[B,N,ceilDiv(Q,X),ceilDiv(KV,Y)]`，1=算/0=跳）；attenMask（必传 nullptr）；blockShape（INT64 `[X,Y]`，Y%128==0，默认 `[128,128]`）；actualSeqLengths/actualSeqLengthsKv（TND 必传，须同时配/不配）；blockTable（必传 nullptr）。
- **输出**：attentionOut（同 query）；softmaxLse（FLOAT，softmaxLseFlag=1 时，TND `[T,N,1]`/BNSD `[B,N,S,1]`）。
- **属性**：qInputLayout/kvInputLayout（TND/BNSD，须一致）；numKeyValueHeads；maskType（仅 0）；scaleValue（默认 1/√D）；innerPrecise（0=fp32/1=fp16 softmax；BF16 仅 0）；blockSize（仅 0）；preTokens/nextTokens（仅 INT_MAX）；softmaxLseFlag（0/1）。
- **约束**：q/k/v dtype 与 layout 一致；D∈{64,128}；blockShapeY 128 倍数；qSeqlen/kvSeqlen 不必整除 blockShape（ceilDiv）；GQA N1≥N2 && N1%N2==0；无 PagedAttention/滑窗/causal mask。
- **workspace**：三缓冲 `[gS|gP|gOTmp|gOUpdate|gSelectNumIdx|gSelectIdx]` + sync buffer（tiling.cpp L605-630）。`selectIdx/selectNumIdx` 由 `Mask2IdxAndCount` 在 workspace 内派生。
- **动态 shape**：支持变长（TND actualSeqLengths）、非对齐（ceilDiv + GEMM tail）。
- **变体**：仅前向；反向为独立算子 `block_sparse_attention_grad`。

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|---|---|---|---|
| Stage0 mask 压缩 | `Mask2IdxAndCount` kernel.h L93-248 | AIV | INT8 mask -> CSR(selectIdx, selectNumIdx) |
| Stage1 QK(bmm1) | kernel.h L616-633 | AIC | Q·K^T（沿 selectIdx gather K）-> set qkReady |
| Stage2 在线 softmax | kernel.h L642-652 | AIV | wait qkReady；running max/sum + rescale -> set softmaxReady |
| Stage3 PV(bmm2) | kernel.h L678-697 | AIC | wait softmaxReady；P·V（沿 selectIdx gather V）-> set pvReady |
| Stage4 rescale O/写回 | kernel.h L717-731 | AIV | wait pvReady；O/=l_i；可选写 LSE |

3 路循环 buffer（`PRE_LAUNCH=2`）；3 个 cross-core flag（qkReady/softmaxReady/pvReady）。

## 5. Tiling parameters

完整表见 `artifacts/block-sparse-attention/tiling_dispatch_tables.md` §B。要点：

| 项 | 值 / 规则 | 来源 |
|---|---|---|
| 基本块 | BASIC_BLOCK_SIZE=128，Q_TILE_CEIL=128，blockShape 默认 [128,128] | kernel_common.hpp L40-84；tiling.cpp L422-424 |
| tile shape | QK `<128,128,128>`；PV L1 `<128,128,256>`、L0 `<128,128,128>` | interface.cpp L68-79 |
| 核间切分轴 | (Q-block × head)；`blockDim=min(aicNum, totalTaskNum)` | tiling.cpp L601 |
| totalTaskNum | `Σ_b ceilDiv(qseqlen_b, blockShapeX) · numHeads` | tiling.cpp L550-553 |
| GQA | groupSize=numHeads/numKvHeads | tiling.cpp L572 |
| AIV 子核（mask 压缩） | avgRowPerSubCore=CeilDiv(B·N·maxQBlockNum, blockDim·2) | tiling.cpp L626-628 |
| workspace | 三缓冲，WORKSPACE_BLOCK_SIZE_DB=131072 | tiling.cpp L605-630 |
| 非对齐 | ceilDiv + GEMM tail（yAcutal） | block_mmad_qk.hpp L200-201 |

## 6. Core partition strategy

- 任务单元 = (batch, head, Q-block)；`for (taskIdx=coreIdx; taskIdx<totalTaskNum; taskIdx+=coreNum)`（kernel.h L447）。
- 每 task 分解：`qSBlockIdx`/`qNBlockIdx`/`qXIdx`（kernel.h L498-501），Q-block 再按 `BASIC_BLOCK_SIZE=128` 子分。
- **KV(S2) 不核间切分**（无 FlashDecoding），核内沿 selectIdx 稀疏遍历。
- GQA：`kvHeadIdx = qNBlockIdx / groupSize`，KV head 跨 groupSize 个 Q head 共享（kernel.h L506-507）。
- tail：qSeqlen 尾块 `xTailNum`、KV 尾块 `yAcutal`；`rowNumRound=AlignUp(rowNum,16)`（kernel.h L503-504,556-564）。
- CV 匹配：AIC:AIV=1:2（MIX_AIC_1_2），vector 核额外承担 mask 压缩；cube/vector 经 3 个 cross-core flag 协调。

## 7. Kernel templates & dispatch

- **12 个 tiling key**（`tilingkey.h` L31-65），base `RFA_BASE_TILING=9000000000000000`。
- 分派维度：**dtype(FP16/BF16) × Q/KV-layout(TND-TND/BNSD-BNSD) × softmax-precision(FLOATSM/HALFSM) × softmaxLseFlag(NONE/OUT_ONLY)**。BF16 仅配 FLOATSM；HALFSM 仅 FP16。6 合法组合 × 2(LSE) = 12。
- **D(64/128) 不进 key**（仅校验）；**blockShape 不进 key**；maskType==3 / pagedCache 为死代码。
- 分派机制：预处理 `#if TILING_KEY_VAR == <KEY>`（block_sparse_attention.cpp L29-94）-> `BlockSparse::BlockSparseAttentionInfer<InputDtype, SoftmaxDtype, LseMode, QueryLayout, KvCacheLayout>`。
- 可迁移：dtype/precision/layout 分支 -> Triton `if constexpr`/独立 kernel；tiling-key 选择规则本身是 CANN 二进制注册机制（不迁移），但其判别条件（dtype/layout/innerPrecise/lseFlag）可迁移为 constexpr meta。

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|---|---|---|---|---|
| mask 压缩 | Mask2IdxAndCount（DataCopyPad/Cast/Compare/GatherMask，AIV） | 稠密 mask->CSR 索引 | host 预处理 / argwhere kernel -> selectIdx | rewrite |
| QK gather | block_mmad_qk 沿 selectIdx copyGmToL1B | 仅算选中块 | `idx=tl.load(selectIdx); tl.dot(q, k[idx])` | rewrite |
| 在线 softmax | block_epilogue_online_softmax（BlockReduceMax/Sum） | running max/sum rescale | `tl.max/sum` + `exp` rescale | transferable |
| PV gather | block_mmad_pv 沿 selectIdx | 仅算选中块 | 间接 `tl.load` V + `tl.dot` | rewrite |
| rescale/写回 | block_epilogue_rescale_o | O/=l_i，LSE | `acc/l_i`，`tl.store` | transferable |
| 跨核协调 | CrossCoreSetFlag/WaitFlag（3 event） | cube↔vector 依赖 | 不迁移；tile 依赖 | CANN-specific |
| 流水 | PreloadPipeline 3-slot（PRE_LAUNCH=2） | 重叠搬运/计算 | 不迁移；tile 复用 | CANN-specific |
| 搬运 | MTE DataCopy GM↔L1/L0/UB、ND↔NZ | 物理搬运/格式 | 不迁移；`tl.load/store`+layout | CANN-specific |
| 缓冲 | TQue/LocalTensor/BufferID（UB/L1/L0） | 显式缓冲管理 | 不迁移；Triton 虚拟内存 | CANN-specific |
| 算力配比 | KERNEL_TYPE_MIX_AIC_1_2 | CV 1:2 | 不迁移 | CANN-specific |

## 9. CANN -> Triton-Ascend migration classification

完整表见 `artifacts/block-sparse-attention/migration-table.md`。汇总：
- **directly transferable**：块稀疏公式、在线 FlashSoftmax rescale、scale_value、FP32 累加、GQA groupSize、softmaxLse 输出、blockSparseMask 语义、ceilDiv tail、blockShapeY 对齐。
- **rewrite**：mask->CSR 压缩、稀疏 KV gather、全 0 行跳过、core id->program_id、BLOCK_M/tile shape constexpr、QK/PV mmad->tl.dot、innerPrecise FP16 路径、12 tiling key 分派、4 阶段流水。
- **CANN-specific（不迁移）**：MTE DataCopy、CrossCore flag、PreloadPipeline 3-slot、TQue/LocalTensor/BufferID、ND↔NZ、KERNEL_TYPE_MIX_AIC_1_2、GatherMask/CompareScalar 原语。
- **needs verification**：tiling 数值、UB/L1 预算、maskType=3 死代码、innerPrecise=4 未实现、PAGED_CACHE_FLAG/blockTable、scaleValue null 默认、LSE 表达式。

## 10. Triton-Ascend skeleton

参考 `artifacts/block-sparse-attention/block-sparse-attention-triton-skeleton.py`。
- 表达：mask 压缩为 selectIdx（host/独立 kernel 预处理）-> attention kernel 沿 selectIdx gather K/V -> 在线 softmax（FP32/Fp16 constexpr 分支）-> GQA -> 可选 LSE。
- `reproducibility: pseudocode`。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `kernelwiki/sources/ops-transformer/block-sparse-attention.md` | source | ops-transformer-block-sparse-attention | - | - | - |
| `kernelwiki/wiki/kernels/block-sparse-attention.md` | kernel | kernel-block-sparse-attention | ops-transformer-block-sparse-attention | inferred | pseudocode |
| `kernelwiki/wiki/techniques/block-sparse-mask-gather.md` | technique | technique-block-sparse-mask-gather | ops-transformer-block-sparse-attention | inferred | snippet |
| `kernelwiki/artifacts/block-sparse-attention/source_file_inventory.md` | artifact | - | - | - | - |
| `kernelwiki/artifacts/block-sparse-attention/tiling_dispatch_tables.md` | artifact | - | - | - | - |
| `kernelwiki/artifacts/block-sparse-attention/migration-table.md` | artifact | - | - | - | - |
| `kernelwiki/artifacts/block-sparse-attention/block-sparse-attention-triton-skeleton.py` | artifact | - | - | - | pseudocode |
| `kernelwiki/artifacts/block-sparse-attention/PROVENANCE.yaml` | provenance | - | - | - | - |

## 12. Validation & retrieval

- `validate.py`：**`OK: KernelWiki validation passed`**（exit 0）。无 missing PROVENANCE、无 sha256 mismatch、无 unresolved sources、无 unknown vocab。（运行期间曾出现无关的 `artifacts/incre-flash-attention/` sha256 mismatch，属另一未跟踪的并行入库 bundle，非本次产物，后续复跑已消失。）
- `generate-indices.py`：成功生成 by-problem/by-technique/by-hardware-feature/by-repo/by-kernel-type 索引。
- `query.py block_sparse_attention --compact`：3 命中（technique-block-sparse-mask-gather、kernel-block-sparse-attention、ops-transformer-block-sparse-attention）。
- `query.py attention tiling --compact`：10 命中，含 kernel-block-sparse-attention。
- `query.py attention mask --compact`：10 命中，含 kernel-block-sparse-attention + technique-block-sparse-mask-gather。
- `query.py attention triton ascend --compact`：10 命中（检索正常）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数具体数值（blockStackNum、sBase 等） | source-reported，最优值需硬件实测 |
| UB/L1 字节预算 / workspace 占用 | 需硬件实测 |
| mask 压缩并行策略 / GatherMask 粒度 | CANN vector 核实现，Triton 侧等价策略待验证 |
| maskType=3 causal 分支 | 死代码（docs 仅支持 0），doc/code 不一致 |
| innerPrecise=4 (LOW_HIGH_MIXED) | host 枚举有，无 kernel tiling key 路径 |
| PAGED_CACHE_FLAG=false 但 blockTable 仍传入 GEMM | paged cache 未支持，blockTable 是否消费待确认 |
| scaleValue attr null -> 1/√D 默认 | 取决于框架 null 指针语义 |
| softmaxLse 表达式 | lse = m_i + ln(l_i) 还是 max+ln(sum)，需对齐 CANN rescale-O 输出 |
| 变长 selectIdx 在 Triton 静态 tile 下的上界对齐 | 需实测选最优 padding/对齐 |

## 14. Modified / created files

创建（均为本次入库产物）：
- `kernelwiki/sources/ops-transformer/block-sparse-attention.md` - 证据来源页（路径/SHA/关键区域摘要）
- `kernelwiki/wiki/kernels/block-sparse-attention.md` - 算子知识页（语义/流水/tiling/变体/agent actions）
- `kernelwiki/wiki/techniques/block-sparse-mask-gather.md` - 可复用方法页（mask->CSR 压缩 + 稀疏 KV gather）
- `kernelwiki/artifacts/block-sparse-attention/source_file_inventory.md` - 源文件清单
- `kernelwiki/artifacts/block-sparse-attention/tiling_dispatch_tables.md` - tiling/dispatch/分核/Triton 映射表
- `kernelwiki/artifacts/block-sparse-attention/migration-table.md` - 四分类迁移表
- `kernelwiki/artifacts/block-sparse-attention/block-sparse-attention-triton-skeleton.py` - Triton 伪代码骨架
- `kernelwiki/artifacts/block-sparse-attention/PROVENANCE.yaml` - 衍生产物溯源（scaffold 生成 + update-provenance 注册 4 文件 sha256）
- `reports/block-sparse-attention-ingest-report.md` - 本报告

未编辑任何受控词汇表（tags.yaml/schemas.yaml 等）或既有 KB 页。

## Quality checklist

- [x] 每个 `sources:` 解析到 source page id；每个 wiki 页引用来源
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 tags.yaml 精确值
- [x] `validate.py` 通过
- [x] `performance_claims: []`
- [x] `confidence: inferred`；`reproducibility: pseudocode`/`snippet`
- [x] 无 "runs/verified/benchmarked" 断言
- [x] CANN 专有机制未作为 Triton 指导（均标 CANN-specific/不迁移）
- [x] 报告列出全部创建文件
