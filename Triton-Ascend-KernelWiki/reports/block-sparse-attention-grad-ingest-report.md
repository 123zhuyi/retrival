# block_sparse_attention_grad CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/`（repo root: `.../Triton-Ascend-KernelWiki`） | source-confirmed |
| OPERATOR_NAME | `block_sparse_attention_grad`（反向） | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred（CANN 仓库 `attention/` 分类；def.cpp 注册为 attention；tags.yaml#kernel_types 含 `attention`） |
| VARIANTS | `backward`（前向 `block_sparse_attention` 已单独入库，见 `reports/block-sparse-attention-ingest-report.md`） | inferred |
| mode | static_ingest | - |

> 前向 `block_sparse_attention`（代号 RFA）的入库报告见 `reports/block-sparse-attention-ingest-report.md`。本反向算子共享前向 `attn_infra` 模板库，但反向专用件不同，且不复用前向的 CSR mask 压缩。

## 2. Source file inventory

算子目录 `attention/block_sparse_attention_grad/`（commit `d28d8f9...`）。核心文件：

| 文件（相对算子目录） | role | centrality | 关键符号 / 行区域 |
|---|---|---|---|
| `docs/aclnnBlockSparseAttentionGrad.md` | doc/api | core | 反向公式 L19-71；参数 L91-320；产品支持 L5-14 |
| `op_host/block_sparse_attention_grad_def.cpp` | host/def | core | 输入 L19-62；输出 dq/dk/dv L63-74；属性 L76-82；AICore L84-85 |
| `op_host/block_sparse_attention_grad_tiling.cpp` | host/tiling | core | GenerateTilingKey(4 key) L497-514；CalculateTaskSplit L369-420；CalculateWorkSpace L422-447 |
| `op_kernel/block_sparse_attention_grad_kernel.h` | kernel | core | `BlockSparseAttentionGradKernel` L61；mask 门控扫描 L386-389,604-607；3 flag L39-41 |
| `op_kernel/block_sparse_attention_grad_interface.h` | kernel | core | 3 cube GEMM 分组 L48/63/78；模板实例化 L123-125 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_simply_softmax.hpp` | template/epilogue | core | 重算 P=exp(S·scale−Lse) L335-393；dS=P·(dP−D) L402-444 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_softmaxgrad.hpp` | template/epilogue | core | D=Σ(dout·out) L208-297 |
| `op_kernel/attn_infra/epilogue/block/block_epilogue_post.hpp` | template/epilogue | core | rescale+写回 L335-411 |
| `op_kernel/attn_infra/gemm/block/block_mmad_bsag2.hpp` | template/gemm | core | 共享 BlockMmad（3 组 cube）；AtomicAdd L149-164 |

完整清单（含行数、SHA-256、attn_infra 全部头）见 `artifacts/block-sparse-attention-grad/source_file_inventory.md`。

## 3. Operator semantics

- **反向公式**（docs L19-71）：`softmax_grad=softmaxGrad(dOut,attentionOut)`；`dP=dOut·V^T`；`dS=P·(dP−softmax_grad)`；`dV=P^T·dOut`；`dQ=(dS·K)·scale`；`dK=(dS^T·Q)·scale`。
- **P 重算**：`P=exp(S·scale−Lse)`，用前向保存 `softmaxLse`；`attentionOut` 用于 softmax-grad 项 `D=Σ(dY·O)`。S/P 均反向重算，不从前向传入。
- **输入**：dout/query/key/value/attentionOut/softmaxLse（必选；softmaxLse=FLOAT）；blockSparseMask（必选，BOOL/UINT8 [needs_verification]）；attenMask（nullptr）；blockShape；actualSeqLengths/actualSeqLengthsKv（TND 必传）。
- **输出**：dq/dk/dv（FP16/BF16）。
- **属性**：qInputLayout/kvInputLayout（TND/BNSD，须一致）；numKeyValueHeads；maskType（仅 0）；scaleValue；preTokens/nextTokens（仅 INT_MAX）。无 innerPrecise、无 softmaxLseFlag（前向专用）。
- **约束**：headdim=128；GQA N1≥N2 && N1%N2==0（README L367 称当前仅 BNSD+MHA [needs_verification]）；无 PagedAttention/滑窗/causal。
- **workspace**（FP32）：`[gS(重算S) | gDp | gDq | gDk | gDv | sftmg]`（L422-447）；无 selectIdx/selectNumIdx。
- **变体**：仅反向；前向为 `block_sparse_attention`。

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|---|---|---|---|
| Stage1 cube | kernel.h L405-422 | AIC | S=Q·K^T + dP=dOut·V^T -> set CUBE2VEC(7) |
| Stage2 vec | kernel.h L597-675 | AIV | wait CUBE2VEC；D=Σ(dY·O)；重算 P；dS=P·(dP−D) -> set VEC2CUBE(8) |
| Stage3 cube | kernel.h L423-511 | AIC | wait VEC2CUBE；dQ=dS·K、dK=dS^T·Q、dV=P^T·dOut（AtomicAdd）-> set CUBE2POST(9) |
| Stage4 vec | kernel.h L515-531 | AIV | wait CUBE2POST+SyncAll；rescale(dQ/dK*=scale)+写回 |

1-deep `preTaskInfo` delay；3 flag CUBE2VEC/VEC2CUBE/CUBE2POST。

## 5. Tiling parameters

完整表见 `artifacts/block-sparse-attention-grad/tiling_dispatch_tables.md` §B。要点：

| 项 | 值 / 规则 | 来源 |
|---|---|---|
| 基本块 | BASIC_BLOCK_SIZE=128，basicQBlockSize=basicKVBlockSize=128 | tiling.cpp L68,467-468 |
| 核间切分轴 | (Q-block × head)；blockDim=min(aicNum, totalTaskNum) | tiling.cpp L418 |
| totalTaskNum | `Σ_b ceilDiv(qseqlen_b, blockShapeX)·numHeads` | tiling.cpp L404-405 |
| 每核任务描述 | beginBatch/beginHead/beginQSeqOffset/preQSeqLengths/preKVSeqLengths[64] | tiling.h L65-69 |
| GQA | groupSize=numHeads/kvHeads | tiling.cpp L217-221 |
| workspace | FP32：gS/gDp/gDq/gDk/gDv/sftmg；无 selectIdx | tiling.cpp L422-447 |
| 非对齐 | ceilDiv + GEMM tail | tiling.cpp L80-87 |

## 6. Core partition strategy

- 任务单元 = (batch, head, Q-block)；每核 `taskNumPerCore + (tailTaskNum > coreIdx ? 1 : 0)`（kernel.h L320）。
- 遍历序：TND `Batch->SeqBlock->Head`；BNSD `Batch->Head->SeqBlock`（tiling.cpp L323-355）。
- **KV(S2) 不核间切分**（无 FlashDecoding），核内沿 KV 块遍历。
- GQA：KV head 每 groupSize 个 Q head 推进（kernel.h L258-260）。
- tail：qSeqlen 尾块、KV 尾块；`rowNumRound=AlignUp(rowNum,16)`。
- CV 匹配：1 cube:2 vec（暗示，入口未显式声明 [needs_verification]）；3 cross-core flag 协调；dQ/dK/dV AtomicAdd 跨块累加。

## 7. Kernel templates & dispatch

- **4 个 tiling key**（`block_sparse_attention_grad.cpp` L44-60）：0=FP16+TND、1=FP16+BNSD、10=BF16+TND、11=BF16+BNSD。仅 dtype×layout。**抛弃**前向 12-key RFA 方案。
- 无 innerPrecise/softmaxLseFlag/maskType/pagedCache 轴；反向 softmax 恒 FP32；Lse 恒消费。
- 分派：`TILING_KEY_IS(0|1|10|11)` -> `BSA::BlockSparseAttentionGradInfer<Dtype, InputLayout>`。
- 3 组 cube GEMM 复用同一 `MmadAtlasA2SBSAG2` 策略（`block_mmad_bsag2.hpp`），Cube2/Cube3 `isAtomicAdd=true`。
- 可迁移：dtype/layout 分支 -> `if constexpr`；tiling-key 选择规则是 CANN 二进制注册（不迁移），判别条件可迁移为 constexpr meta。

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|---|---|---|---|---|
| mask 门控扫描 | if(mask.GetValue()) skip（cube+vec 均扫） | 跳过未选中 KV 块 | `tl.load(mask); tl.where/continue` | rewrite |
| 重算 P | SimpltSoftmax（exp(S·scale−Lse)） | 用前向 Lse 恢复 P | `tl.exp(tl.dot(q,k)*scale - lse)` | transferable |
| softmax-grad D | SoftmaxGrad（Σ(dY·O)） | 归一化项 | `tl.sum(dy*o, axis=-1)` | transferable |
| 3 cube GEMM | block_mmad_bsag2（S&dP/dQ/dK&dV） | dQ/dK/dV | `tl.dot` 三组 | rewrite |
| dQ/dK/dV 累加 | AtomicAdd（Cube2/3） | 跨 KV 块/跨程序 | `tl.atomic_add`/per-program accum | rewrite |
| rescale/写回 | BlockPost（dQ/dK*=scale，dV 不） | 缩放+cast+写回 | `dq*=scale; dk*=scale` | transferable |
| 跨核协调 | CrossCore flag（7/8/9） | cube↔vec 依赖 | 不迁移；tile 依赖 | CANN-specific |
| 流水 | preTaskInfo 1-deep delay | 重叠 softmax | 不迁移；tile 复用 | CANN-specific |
| 搬运 | MTE DataCopy / ND↔NZ | 物理搬运 | 不迁移；`tl.load/store`+layout | CANN-specific |
| 缓冲 | TQue/LocalTensor/BufferID | 显式缓冲 | 不迁移 | CANN-specific |
| softmax-grad intrinsic | SoftmaxGradFront/SoftMaxGradTilingFunc | 高性能 reduction | 不迁移；手写 reduction | CANN-specific |

## 9. CANN -> Triton-Ascend migration classification

完整表见 `artifacts/block-sparse-attention-grad/migration-table.md`。汇总：
- **directly transferable**：反向公式、softmax-grad D、dS=P·(dP−D)、重算 P、scale（dQ/dK 乘、dV 不）、FP32 累加、GQA、blockSparseMask 语义、ceilDiv tail。
- **rewrite**：mask 门控稠密扫描、重算 P、3 cube GEMM、dQ/dK/dV AtomicAdd、core id->program_id、4 tiling key 分派、4 阶段流水、BLOCK_M constexpr。
- **CANN-specific（不迁移）**：MTE DataCopy、CrossCore flag(7/8/9)、preTaskInfo delay、TQue/LocalTensor/BufferID、ND↔NZ、AtomicAdd(CANN)、SoftmaxGradFront/SoftMaxGradTilingFunc。
- **needs verification**：tiling 数值、UB/L1 预算、mask dtype 不一致、headdim 强制、TND+GQA 支持、任务类型宏缺失、scaleValue=0.0 默认、SBSAG1/SBSAG3 未用、mask 门控 vs CSR 取舍。

## 10. Triton-Ascend skeleton

参考 `artifacts/block-sparse-attention-grad/block-sparse-attention-grad-triton-skeleton.py`。
- 表达：mask 门控扫描（if mask==0 continue）-> 重算 P=exp(S·scale−lse) -> D=Σ(dY·O) -> dS=P·(dP−D) -> dQ 核内累加 / dK/dV 跨程序 `tl.atomic_add` -> rescale(dQ/dK*=scale)。
- `reproducibility: pseudocode`。未编译、未 NPU 验证、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|---|---|---|---|---|---|
| `kernelwiki/sources/ops-transformer/block-sparse-attention-grad.md` | source | ops-transformer-block-sparse-attention-grad | - | - | - |
| `kernelwiki/wiki/kernels/block-sparse-attention-grad.md` | kernel | kernel-block-sparse-attention-grad | ops-transformer-block-sparse-attention-grad | inferred | pseudocode |
| `kernelwiki/artifacts/block-sparse-attention-grad/source_file_inventory.md` | artifact | - | - | - | - |
| `kernelwiki/artifacts/block-sparse-attention-grad/tiling_dispatch_tables.md` | artifact | - | - | - | - |
| `kernelwiki/artifacts/block-sparse-attention-grad/migration-table.md` | artifact | - | - | - | - |
| `kernelwiki/artifacts/block-sparse-attention-grad/block-sparse-attention-grad-triton-skeleton.py` | artifact | - | - | - | pseudocode |
| `kernelwiki/artifacts/block-sparse-attention-grad/PROVENANCE.yaml` | provenance | - | - | - | - |

未新建 technique 页（反向 mask 门控扫描作为前向 [[technique-block-sparse-mask-gather]] 的对照在 kernel 页 §3 描述；反向重算 P 引用既有 `technique-flash-attention-backward-recompute`）。

## 12. Validation & retrieval

- `validate.py`：**`OK: KernelWiki validation passed`**（exit 0）。
- `generate-indices.py`：成功生成 5 个索引。
- `query.py block_sparse_attention_grad --compact`：3 命中（source、前向 source、kernel）。
- `query.py attention backward --compact`：含 kernel-block-sparse-attention-grad + technique-flash-attention-backward-recompute + kernel-block-sparse-attention。
- `query.py attention tiling --compact`：含 kernel-block-sparse-attention-grad。
- `query.py attention triton ascend --compact`：检索正常（10 命中）。

## 13. Unverified / needs-hardware items

| 项 | 原因 |
|---|---|
| tiling 参数具体数值（postUbBaseSize 等） | source-reported，需硬件实测 |
| UB/L1 字节预算 / workspace 占用 | 需硬件实测 |
| mask dtype（def=UINT8 / README=BOOL / kernel=bool） | 三处不一致，需确认 |
| headdim=128 强制 | README 说仅 128，代码定义未强制 |
| TND+GQA 支持 | README L367 称仅 BNSD+MHA，与参数表/代码不一致 |
| 任务类型宏缺失 | 入口未显式 KERNEL_TYPE_MIX_AIC_1_2，是否 host 设置待确认 |
| scaleValue=0.0 默认 | null 才取 1/√D，0.0 是否归一化待确认 |
| SBSAG1/SBSAG3 未用 | 声明但无 BlockMmad 特化 |
| mask 门控 vs CSR gather 取舍 | 反向选 branch-on-mask，稀疏度阈值需实测 |
| AtomicAdd 数值精度 | dK/dV 跨程序 atomic 累加，精度需验证 |
| dV 是否乘 scale | 代码 dV 不乘（BlockPost L371-374），需对照公式确认 |

## 14. Modified / created files

创建（均为本次反向入库产物）：
- `kernelwiki/sources/ops-transformer/block-sparse-attention-grad.md` - 反向证据来源页
- `kernelwiki/wiki/kernels/block-sparse-attention-grad.md` - 反向算子知识页（含 mask 门控 vs CSR 对照、重算 P、4 阶段流水）
- `kernelwiki/artifacts/block-sparse-attention-grad/source_file_inventory.md` - 源文件清单
- `kernelwiki/artifacts/block-sparse-attention-grad/tiling_dispatch_tables.md` - tiling/dispatch/分核/Triton 映射表
- `kernelwiki/artifacts/block-sparse-attention-grad/migration-table.md` - 四分类迁移表
- `kernelwiki/artifacts/block-sparse-attention-grad/block-sparse-attention-grad-triton-skeleton.py` - Triton 伪代码骨架
- `kernelwiki/artifacts/block-sparse-attention-grad/PROVENANCE.yaml` - 衍生产物溯源（4 文件 sha256）
- `reports/block-sparse-attention-grad-ingest-report.md` - 本报告

编辑：
- `kernelwiki/wiki/kernels/block-sparse-attention-grad.md` 的 `related:` 增补 `technique-flash-attention-backward-recompute`（既有并行入库的技术页，反向重算 P 相关）。

未编辑任何受控词汇表（tags.yaml/schemas.yaml 等）或既有 KB 页（除本算子新建页）。

## Quality checklist

- [x] 每个 `sources:` 解析到 source page id；每个 wiki 页引用来源
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 均为 tags.yaml 精确值
- [x] `validate.py` 通过
- [x] `performance_claims: []`
- [x] `confidence: inferred`；`reproducibility: pseudocode`
- [x] 无 "runs/verified/benchmarked" 断言
- [x] CANN 专有机制未作为 Triton 指导（均标 CANN-specific/不迁移）
- [x] 报告列出全部创建/编辑文件
