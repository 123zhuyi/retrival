# nsa_compress_with_cache - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/nsa_compress_with_cache/`。`fixture: true`。无 arch 子目录。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 111 | doc | core | 功能/压缩流程 L17-23；公式 L25-29；参数 L31-91；约束 L93-105 | `82d127c308cf4731...` |
| `docs/aclnnNsaCompressWithCache.md` | 528 | doc/api | core | aclnn API 语义 | `69ca072adafe81cc...` |
| `op_host/nsa_compress_with_cache_def.cpp` | 119 | host/def | core | `NsaCompressWithCache` OpDef：输入 L21-51；输出 L52-56；属性 L57-61；AICore config(910b/910_93 L62-63 + kirinX90/9030 L65-67) | `b48e0085036c4ab4...` |
| `op_host/nsa_compress_with_cache_infershape.cpp` | 43 | host | core | infershape | `b97562b79a5e0fe3...` |
| `op_host/nsa_compress_with_cache_tiling.cpp` | 110 | host/tiling | core | tiling 主逻辑 | `61d89af0cc393864...` |
| `op_host/nsa_compress_with_cache_tiling.h` | 52 | host/tiling | core | `NsaCompressWithCacheTilingData`（kvCacheSize/weightSize/headNum/headDim/batchSize/compressBlockSize/compressStride/coresNumPerCompress/headsNumPerCore/tokenNumPerTile/bufferNum/coreStartBatchIdxList/coreStartHeadIdxList） | `d5b233357b2c79d7...` |
| `op_host/nsa_compress_with_cache_tiling_common.h` | 24 | host/tiling | core | 公共常量 | `f154568725083b14...` |
| `op_host/nsa_compress_with_cache_tiling_general.cpp` | 548 | host/tiling | core | 通用 tiling 计算 | `952b8deeb37c8b36...` |
| `op_kernel/nsa_compress_with_cache.cpp` | 486 | kernel | core | `KernelNsaCompressWithCacheBase`：InitBase L38-64（blockIdx/coreStartHeadIdx/startBatchIdx/ping-pong buffer）；InitActSeqLen L66-78（检查 compress 条件）；InitWeight L80-91（weight cast to FP32）；Process L127+（压缩循环） | `725b2fecb646e6da...` |
| `tests/utest/op_kernel/ts_nsa_compress_with_cache_kernel.cpp` | 441 | test | indirect | kernel UT | `799e554446b38b01...` |

另含 `op_api/`(4)、`tests/comm/`(8) 等辅助文件。