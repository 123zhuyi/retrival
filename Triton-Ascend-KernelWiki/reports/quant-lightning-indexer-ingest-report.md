# quant_lightning_indexer CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer/attention/quant_lightning_indexer/` | source-confirmed |
| KERNELWIKI_ROOT | `/home/w00836812/shixi_test02/Triton-Ascend-KernelWiki/kernelwiki` | source-confirmed |
| OPERATOR_NAME | `quant_lightning_indexer` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred (CANN ops-transformer/attention 目录，tags.yaml#kernel_types 含 attention) |
| VARIANTS | `inference` | source-confirmed (推理场景 SFA 前处理，README.md L13) |
| mode | `static_ingest` | - |

## 2. Source file inventory

完整表见 `artifacts/quant-lightning-indexer/source-file-inventory.md`（~20 文件，按 op_host/op_kernel/op_api/tests 分角色与中心度）。

核心文件（相对算子目录）：

| path | role | centrality | key symbols |
|------|------|-----------|-------------|
| `op_host/quant_lightning_indexer_def.cpp` | 定义 | core | OpDef I/O(int8/fp8)、属性、AICore 配置 ascend910b/910_93/950 |
| `op_host/quant_lightning_indexer_tiling.cpp` | tiling | core | QLITilingData、CalcS1S2、CalcCoreNum、PageAttention |
| `op_kernel/quant_lightning_indexer.cpp` | kernel入口 | core | 6 模板参数 dispatch |
| `op_kernel/quant_lightning_indexer_kernel.h` | kernel模板 | core | QLIPreload（QKCompute + IndexCompute + TopK + WriteBack）|

## 3. Operator semantics

- **输入**：query/key（int8 / fp8_e4m3fn/hifloat8）、weights（FP16/BF16）、query_dequant_scale（FP16/FLOAT）、key_dequant_scale（FP16/FLOAT）、actual_seq_lengths_query/key（INT32）、block_table（INT32）
- **输出**：sparse_indices（INT32）
- **属性**：query_quant_mode(0 per-token-head)、key_quant_mode(0)、layout_query(BSND/TND)、layout_key(BSND/TND/PA_BSND)、sparse_count(2048)、sparse_mode(0/3)、pre_tokens(INT64_MAX)、next_tokens(INT64_MAX)
- **核心公式**：`out = Top_k{ W @ [ ReLU((Scale_Q · Scale_K^T) · (Q^Quant @ K^Quant^T)) ] }`
- **约束**：D=128 固定；N2=1；G=64(arch32)/24,64(arch35)；sparse_count[1,2048]；PA block_size 16 倍数 ≤1024

## 4. Stage / pipeline table

| Stage | Entry | Unit | Op |
|-------|-------|------|----|
| Preload | QLIPreload::Init | AIC+AIV | 输入加载、反量化系数准备 |
| Cube | QKCompute | AIC | Q·K^T matmul（反量化 FP32 累加） |
| Vector | IndexCompute | AIV | Scale_Q·Scale_K^T → ReLU → W@加权 |
| Topk | TopK | AIV | topk 索引选择 |
| WriteBack | WriteBack | AIV | sparse_indices 写回 GM |

## 5. Tiling parameters

- D=128 固定（HEAD_DIM_LIMIT=128）
- S2 不切分（整块计算）
- S1 分核：`CalcCoreNum` 按 S1 与 sparse_count 确定核数
- `KERNEL_TYPE_MIX_AIC_1_2`（1 Cube:2 Vector）
- PageAttention：block_table 映射离散 KV 块

## 6. Core partition strategy

- S1 分核：每个 program 处理一个 S1 块
- S2 整块：不跨核切分
- 核间切分按 `usedCoreNum` 分配

## 7. Kernel templates & dispatch

- **单模板** `QLIPreload`，6 模板参数（DT_Q/DT_K/DT_OUT/PAGE_ATTENTION/Q_LAYOUT_T/K_LAYOUT_T）
- arch32 2 组合（int8 × PA_BSND/BSND/TND）
- arch35 6 组合（fp8_e4m3fn/hifloat8 × PA_BSND/BSND/TND）
- 可迁移触发：dtype/layout/PA flag → Triton constexpr 分支
- 不迁移：tiling key bit 编码、REGISTER 机制

## 8. Pipeline analysis

| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |
|-------|---------------|---------|-------------------|---------------|
| 数据读入 | DataCopy GM->UB | 载入 Q/K/weights/scale | `tl.load` | 是 |
| 反量化 | QuantDequant 向量算子 | int8/fp8 → FP32 | `tl.load(..).to(tl.float32)*scale` | 是 |
| Q·K^T | Cube matmul | 相关性 | `tl.dot(q,k.T)` FP32 | 是 |
| Scale·Scale^T | Muls 逐元素乘 | 反量化系数融合 | 乘标量 | 是 |
| ReLU | ReLU 激活 | 过滤负相关 | `tl.maximum(0, x)` | 是 |
| W@加权 | matmul/reduce | 跨 G 轴聚合 | `tl.dot(w, scores)` | 是 |
| topk | TopK API | 稀疏索引 | `tl.topk(x, sparse_count)` | 是 |
| 写回 | DataCopy | 输出 | `tl.store` | 是 |
| 手动同步 | SetFlag/WaitFlag | CV 依赖 | 编译器排流 | 否（CANN-specific） |

## 9. CANN -> Triton-Ascend migration classification

完整表见 `artifacts/quant-lightning-indexer/migration-table.md`。摘要：
- **transferable**：量化相关性公式、反量化语义、ReLU、topk、FP32 累加、因果掩码
- **rewrite**：单模板、CV 分离、核间切分、PA_BSND block_table 映射
- **CANN-specific（不迁移）**：MTE/DataCopy、BufferID/queue/event、手动 CV 时序、TBuf 放置
- **needs_verification**：基本块数值最优性、UB 占用、PA 运行时契约、性能

## 10. Triton-Ascend skeleton

参考 `artifacts/quant-lightning-indexer/quant-lightning-indexer-triton-skeleton.py`。
`reproducibility: pseudocode`。未编译、未 benchmark。

## 11. KB pages written

| Path | type | id | sources | confidence | reproducibility |
|------|------|----|---------|-----------|----------------|
| `kernelwiki/sources/ops-transformer/quant-lightning-indexer.md` | source | ops-transformer-quant-lightning-indexer | - | - | - |
| `kernelwiki/wiki/kernels/quant-lightning-indexer.md` | kernel | kernel-quant-lightning-indexer | [ops-transformer-quant-lightning-indexer] | inferred | pseudocode |
| `kernelwiki/artifacts/quant-lightning-indexer/PROVENANCE.yaml` | provenance | - | - | - | - |
| `kernelwiki/artifacts/quant-lightning-indexer/source-file-inventory.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/quant-lightning-indexer/tiling-dispatch-tables.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/quant-lightning-indexer/pipeline-diagram.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/quant-lightning-indexer/migration-table.md` | approach-notes | - | - | - | - |
| `kernelwiki/artifacts/quant-lightning-indexer/quant-lightning-indexer-triton-skeleton.py` | derived-source | - | - | - | pseudocode |

## 12. Validation & retrieval

- `validate.py`: **OK: KernelWiki validation passed**
- `generate-indices.py`: 生成 5 个索引
- `query.py quant_lightning_indexer --compact`: 预期 2 命中（source 页 + kernel 页）

## 13. Unverified / needs-hardware items

- tiling 基本块数值是否最优 -- 需硬件实测
- UB 占用 -- 需 profile
- PageAttention block_table 运行时契约 -- 需运行时确认
- int8/fp8 精度边界 -- 需 correctness test
- 性能（任何加速比）-- `static_ingest` 无 baseline

## 14. Modified / created files

- `kernelwiki/artifacts/quant-lightning-indexer/PROVENANCE.yaml` — bundle 溯源
- `kernelwiki/artifacts/quant-lightning-indexer/source-file-inventory.md` — S1 源文件清单
- `kernelwiki/artifacts/quant-lightning-indexer/tiling-dispatch-tables.md` — S4/S5 tiling + dispatch 表
- `kernelwiki/artifacts/quant-lightning-indexer/pipeline-diagram.md` — S6 流水抽象
- `kernelwiki/artifacts/quant-lightning-indexer/migration-table.md` — S3 迁移分类
- `kernelwiki/artifacts/quant-lightning-indexer/quant-lightning-indexer-triton-skeleton.py` — S7 Triton skeleton
- `kernelwiki/sources/ops-transformer/quant-lightning-indexer.md` — source 页
- `kernelwiki/wiki/kernels/quant-lightning-indexer.md` — kernel wiki 页
- `reports/quant-lightning-indexer-ingest-report.md` — 本报告
- `kernelwiki/queries/*.md` — 由 generate-indices.py 重新生成

## Quality checklist

- [x] Every `sources:` resolves to a source page id；每个 wiki 页 cite source
- [x] `architectures/hardware_features/techniques/kernel_types/symptoms` 为 tags.yaml 精确值
- [x] `validate.py` 通过
- [x] `performance_claims: []`（无 baseline）
- [x] `confidence: inferred`；`reproducibility: pseudocode`
- [x] 无 "runs/verified/benchmarked" 主张
- [x] CANN 机制（MTE/BufferID/queue/event/手动同步）未作为 Triton 指导
- [x] 报告列出所有创建/编辑文件