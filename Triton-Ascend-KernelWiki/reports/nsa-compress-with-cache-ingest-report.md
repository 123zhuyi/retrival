# nsa_compress_with_cache CANN->Triton-Ascend ingestion report

## 1. Task parameters

| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer` | source-confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | source-confirmed |
| OPERATOR_NAME | `nsa_compress_with_cache` | source-confirmed |
| OPERATOR_TYPE | `attention` | inferred |
| VARIANTS | inference（NSA KV 压缩） | inferred |
| commit | `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` | source-confirmed |
| mode | static_ingest | - |

## 2. Source file inventory

~25 文件，无 arch 子目录。核心：`README.md`(111L)、`docs/aclnnNsaCompressWithCache.md`(528L)、`op_host/nsa_compress_with_cache_def.cpp`(119L)、`op_host/nsa_compress_with_cache_tiling.h`(52L)、`op_kernel/nsa_compress_with_cache.cpp`(486L)。完整清单（含 SHA-256）见 `artifacts/nsa-compress-with-cache/source_file_inventory.md`。

## 3. Operator semantics

- **功能**：NSA 推理 KV 压缩。当 batch 的 seqLen 满足 `(len-compressBlockSize)%stride==0`，将后 compressBlockSize 个 token 用 weight 加权求和后通过 slot_mapping 写回 cache（README L17-23）。
- **公式**：`outputCache[slot_mapping[i]] = Σ(input[compressIdx*stride : compressIdx*stride+compressBlockSize] × weight[:])`（README L28-29）。
- **输入**：input/weight/slot_mapping/output_cache（REQUIRED，FP16/BF16，ND）+ act_seq_len/block_table（OPTIONAL）（def.cpp L21-51）。
- **输出**：output_cache（REQUIRED，原地更新）。
- **属性**：compress_block_size/compress_stride/act_seq_len_type/page_block_size（必选 int）。
- **约束**：compressBlockSize/stride 16 倍数，≤64；headDim≤256 16 倍数；headNum≤64；pageBlockSize=64/128。不支持 950/310P/训练系列（README L93-105）。
- **架构**：A2/A3(FP16/BF16) + KirinX90/Kirin9030(仅 FP16)（def.cpp L62-67）。

## 4. Stage / pipeline table

| 阶段 | 操作 | 来源 |
|---|---|---|
| 1. InitBase | 读 tiling data，分核，分配 ping-pong buffer | kernel L38-64 |
| 2. InitActSeqLen | 遍历 batch，检查压缩条件 | kernel L66-78 |
| 3. InitWeight | weight 搬入 UB，Cast to FP32 | kernel L80-91 |
| 4. Compress | 滑动窗口加权求和 → scatter store | Process |

## 5. Tiling

分核：按 head×batch 切分（`coreStartBatchIdxList`/`coreStartHeadIdxList`）。核内：`headsNumPerCore`、`tokenNumPerTile`/`tokenRepeatCount`、`bufferNum` ping-pong。

## 6. Core partition

head×batch 双轴切分，每核从 `coreStartBatchIdxList[blockIdx]` 到 `[blockIdx+1]` 遍历 batch，每 batch 处理 `coreStartHeadIdxList[blockIdx]` 到 `[blockIdx+1]` 的 head。

## 7. Kernel templates & dispatch

单模板（模板参数由 tiling data 传入）。`weightInitFlag` 控制 weight 初始化方式。

## 8. Pipeline analysis

滑动窗口 offset 计算 → 加权求和 → scatter 写回。ping-pong buffer 为 CANN 专有。

## 9. Migration classification

完整表见 `artifacts/nsa-compress-with-cache/migration-table.md`。核心 transferable：压缩公式、weight broadcast、slot_mapping scatter、FP32 中间。rewrite：分核→program_id、constexpr meta。CANN-specific：ping-pong buffer、GetBlockIdx。

## 10. Triton-Ascend skeleton

见 `artifacts/nsa-compress-with-cache/nsa-compress-with-cache-triton-skeleton.py`。`reproducibility: pseudocode`。

## 11. KB pages written

| Path | type | id |
|---|---|---|
| `sources/ops-transformer/nsa-compress-with-cache.md` | source | `ops-transformer-nsa-compress-with-cache` |
| `wiki/kernels/nsa-compress-with-cache.md` | kernel | `kernel-nsa-compress-with-cache` |
| `artifacts/nsa-compress-with-cache/`(6 文件) | bundle | - |

## 12. Validation & retrieval

```
validate.py        -> OK
generate-indices   -> OK
query nsa compress with cache -> #1 hit kernel-nsa-compress-with-cache ✓
query nsa compress -> 10 hits（4 个 NSA kernel 页完整配对）✓
```

注：`query nsa_compress_with_cache`（下划线）只命中 source 页（title 精确匹配），kernel 页用空格 `nsa compress with cache` 命中。这是 `query.py` 的文本评分行为，非 KB 质量问题。

## 13. Unverified

tiling 参数最优值、weightInitFlag 含义。

## 14. Created files

**新建**：`sources/ops-transformer/nsa-compress-with-cache.md`、`wiki/kernels/nsa-compress-with-cache.md`、`artifacts/nsa-compress-with-cache/`(6 文件)、`reports/nsa-compress-with-cache-ingest-report.md`。