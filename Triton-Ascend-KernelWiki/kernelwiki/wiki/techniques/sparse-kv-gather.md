---
id: technique-sparse-kv-gather
title: Sparse KV cache gather and consolidation for SparseFlashAttention on Ascend NPU
type: technique
tags: [sparse_attention, kv_cache, gather, dma, page_attention, memory_access]
techniques: [sparse_kv_gather]
hardware_features: [unified_buffer, l1_buffer, l2_cache]
architectures: [Ascend910B1, Ascend910_93]
kernel_types: [attention]
operators: [sparse_flash_attention]
symptoms: [performance_bottleneck]
confidence: inferred
reproducibility: snippet
sources: [ops-transformer-sparse-flash-attention]
prerequisites: [了解FlashAttention KV搬运流程, PageAttention/块表基本概念, Ascend ND/NZ格式与MTE搬运基本概念]
related: [technique-sparse-attention-tiling, technique-sparse-attention-pipeline, technique-flash-attention-pipeline]
indicators: [indicator-01, indicator-04, indicator-05, indicator-07]
---

# Sparse KV cache gather and consolidation

SparseAttention 只计算"关键部分"的 QK 和 PV，但关键 KV 的索引由外部算法（如 lightning_indexer）给出，存在大量离散访存。SFA 的优化核心是将离散的 KV block 搬运聚合为连续数据，减少 MTE 次数和搬运开销。

## 1. 问题背景

标准 FlashAttention 中 KV 序列在 Global Memory (GM) 中连续存储，DataCopy 可一次搬运任意连续范围。SparseAttention 的 KV 由稀疏索引（topK sparse_indices）映射，映射后的物理位置在 GM 中 **不连续**，导致：

- 若 Cube 直接搬运，每次 DataCopy 只能搬运少量连续数据，MTE 利用率低。
- 离散 KV 之间存在大量不可预知的间隔（stride），需要额外的偏移计算。

SFA 的解决思路是：**在 Vector 阶段（V0 / MergeKv）将离散 KV block 聚合到连续 buffer，再供 Cube 批量读取**。

## 2. MergeKv：离散 KV 合并

V_TEMPLATE 模式下，Vector0 阶段执行 `MergeKv`（`vector_mla.h:986`），核心流程：

```
sparse_indices (topKGm)  ──→  GetRealS2Idx()  ──→  GetKeyGmOffset()
                                   │                        │
                                   ▼                        ▼
                             离散 KV 在 GM 中的偏移 ──→  CopyInKv()
                                                              │
                                                              ▼
                                                      kvMergeGm_ (连续 buffer)
                                                              │
                                                              ▼
                                                     CopyOutMrgeResult()
                                                              │
                                                              ▼
                                                      Cube Mm1/Mm2
```

### 2.1 索引映射：GetRealS2Idx

`GetRealS2Idx`（`vector_mla.h:821-831`）将逻辑 sparse block 索引映射为物理 KV 索引：

```
topkGmIdx = (s2GmOffset + s2Idx * s2BaseSize) / sparseBlockSize
realS2Idx = topKGm[topkGmIdx] * sparseBlockSize + (s2GmOffset + s2Idx * s2BaseSize) % sparseBlockSize
```

- `topKGm` 是 sparse_indices 数组，存放原始 KV 序列中的绝对位置。
- `sparseBlockSize` 是稀疏 block 大小（元素数），固定为 512（SFA MLA 场景）。
- 返回值 `realS2Idx` 是 KV 在原始连续序列中的行号（即 seq 维度索引）。

### 2.2 偏移计算：GetKeyGmOffset

`GetKeyGmOffset`（`vector_mla.h:834-853`）将 `realS2Idx` 转换为 GM 中的实际偏移（element 级）：

- **非 PageAttention 场景**：直接从 `tensorBOffset` 和 `realS2Idx` 计算偏移。
- **PageAttention 场景**（`PAGE_ATTENTION == true`）：通过 `blkTableGm_` 查询物理 block 编号，结合 `kvCacheBlockSize` 和 `kvHeadNum` 计算偏移。

### 2.3 数据搬运：CopyInKv / CopyInSingleKv

`CopyInKv`（`vector_mla.h:896-957`）负责将两个离散的 KV sparse block 从 GM 搬运到 UB：

- **优化路径**：当两个 block 的 stride 满足 `0 ≤ stride < INT32_MAX` 时，合并为一条 `DataCopyPad` 指令（一次 MTE 搬运两个 block，利用 srcStride 参数跳过间隔）。
- **回退路径**：stride 溢出或超尾时，回退为两条独立的 `CopyInSingleKv` 指令（`vector_mla.h:872-893`）。
- 每次搬运同时搬运 **key** 和 **keyRope** 两部分，写入 `kvMergUb_` 和 `ropeMergUb_` 的 ping-pong buffer。

### 2.4 结果写回：CopyOutMrgeResult

`CopyOutMrgeResult`（`vector_mla.h:960-982`）将 UB 中合并后的连续 KV 数据写回 GM 的 `kvMergeGm_` buffer：

- 使用 `DataCopyPad` 将 UB 中连续数据一次性写回 GM。
- 支持 4 路 `kvMergeGm_` buffer（`runInfo.loop % 4`），实现不同 S2 loop 之间的流水。
- 同时写回 key 和 keyRope 两个部分，地址偏移分别为 `512 * 512` 和 `512 * 576` （`headDim=512, headDimRope=64`）。

### 2.5 主循环调度

`MergeKv` 主循环（`vector_mla.h:1013-1030`）以 `2 * sparseBlockSize` 步长遍历 S2 维度：

- 每次处理两个 sparse block，利用 `MTE3_MTE2` 同步信号控制 MTE 流水。
- 不合法索引（`realS2Idx < 0`）时提前触发 `CopyOutMrgeResult` 将已缓存数据落盘。
- 双核切分（`GetSubBlockIdx()`）：S2 方向在双核上均分，各核处理一半 sparse block。

## 3. PageAttention 场景的离散搬运

`DataCopyPA`（`cube_mla.h:72-113`）处理 PageAttention 下 KV 从 GM 到 L1 的离散搬运：

```
物理 block 映射：blockTableGm[ batchIdx * maxBlockNumPerBatch + blockId ]
```

核心逻辑：

1. **block 表查询**：`curS2Idx / blockSize` 获取 block table 索引，`blockTableGm.GetValue()` 获取物理 block 编号。
2. **块内偏移**：`reaminRowCnt = curS2Idx % blockSize` 确定块内起始行。
3. **边界处理**：一次最多搬运一个完整 block 的剩余行数（`copyRowCnt = blockSize - reaminRowCnt`），超尾时截断。
4. **数据排布**：
   - **BNBD**（`BSND/TND`）：`offset = blockId * blockSize * headNum * headDim + n2Idx * headDim + reaminRowCnt * headDim * headNum`
   - **BBH**（默认）：`offset = blockId * blockSize * headNum * headDim + n2Idx * headDim * blockSize + reaminRowCnt * headDim`

`DataCopyPA` 的调用通过 `DataCopyGmNDToL1` 完成实际的 ND→NZ 格式转换搬运。

## 4. ND→NZ 格式转换搬运

`DataCopyGmNDToL1`（`cube_mla.h:47-64`）在数据搬运的同时完成格式转换：

- **GM 存储格式**：ND（row-major，连续的 `n*d` 矩阵）
- **L1 存储格式**：NZ（tiled，`C0=16` 分块）
- 通过 `Nd2NzParams` 配置转换参数：
  - `ndNum`: ND 矩阵个数
  - `nValue`: ND 矩阵行数（即 `rowAct`）
  - `dValue`: ND 矩阵列数（即 `headDim` / `actHeadDim`）
  - `srcDValue`: 行间 stride（等于 `headDim` 或 `headNum * headDim`）
  - `dstNzC0Stride`: L1 中 C0 维 stride（等于 `rowAlign`）
- 一条 `DataCopy` 指令同时完成 **格式转换 + 数据搬运**，减少 MTE 指令数。

适用场景：Q、K、V 从 GM 到 L1 的加载，以及 PageAttention 中 `DataCopyPA` 的底层搬运。

## 5. 对齐约束

`BlockAlign`（`common.h:59-66`）定义了数据对齐规则：

```cpp
int4 场景: (s + 63) / 64 * 64
其他类型: (32 / sizeof(T)) 对齐
```

| 常量 | 值 | 说明 |
|------|-----|------|
| `BYTE_BLOCK` | 32 | Vector 访问的基本单位，所有数据传输地址必须 32B 对齐 |
| `REPEAT_BLOCK_BYTE` | 256 | Vector repeat 指令的基本单位，单次 repeat 最多处理 256B |

对齐约束的影响：

- `copyRowNumAlign` 在 PageAttention 中需要 16B 对齐（如 10×512 尾块对齐到 16×512）。
- `copyRowCnt` 在 `DataCopyPA` 中由 `blockSize` 和 `reaminRowCnt` 共同决定，天然对齐。
- MergeKv 的 `blockLen` 为 `sparseBlockSize * headDim * sizeof(KV_T)`，由于 `sparseBlockSize=512`, `headDim=512`，均为 32B 倍数。

## 6. 输出转置写回

Mm2 输出在 UB 中为连续格式，写回 GM 时需匹配目标 layout 要求。

`Bmm2DataCopyOutTrans`（`vector_mla.h:1244-1255`）处理输出转置：

- **srcStride**：`(columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T))`，跳过 UB 中 padding 列。
- **dstStride**：0，连续写回 GM。
- 写回目标偏移：`attentionOutGm[attenOutOffset + wsMStart * actualColumnCount]`。

对于需要复杂转置的场景（如 `ConfusionTransposeTiling`），通过 `GetConfusionTransposeTiling`（`vector_mla.h:116-117`）配置转置参数。

## 7. 对 Triton 实现的启示

从 SFA 的离散 KV 搬运策略，可以提炼出对 Triton 编程的指导：

1. **离散读取 + 聚合写入**：在 Triton 中使用 `tl.load` 配合稀疏索引做离散读取，使用 `tl.store` 将离散数据合并到连续 buffer。对应 `MergeKv` 的 `CopyInKv → CopyOutMrgeResult` 流程。

2. **对齐约束**：block 大小应为 32B 的倍数。Triton 的 `tl.constexpr` 可预处理对齐参数，避免运行期对齐开销。

3. **PageAttention 模拟**：Triton 中需要模拟 block table 查表逻辑——通过 `tl.load` 读取 block table，序列化映射后作为 `tl.load` 的索引。

4. **格式转换合并**：当 Triton 的 memory layout 与 Ascend 的 ND→NZ 类似时（如 row-major → tiled），应尽量将格式转换与数据搬运合并到一条指令，减少显式 permute 调用。

5. **双 buffer 流水**：MergeKv 使用 ping-pong buffer（`mergeMte3Idx % 2`）和 4 路 `kvMergeGm_` buffer。Triton 中可用 `tl.max_scope` 或显式 double buffer 模式实现类似的效果。

6. **int4 精度**：`BlockAlign` 对 int4 有特殊处理（64B 对齐），因 int4 类型在 Ascend 上 2 个元素合 1 字节。Triton 中使用 int4 加载时也需注意对齐倍数。

## 8. 关键源码引用

| 函数 | 文件 | 行号 | 功能 |
|------|------|------|------|
| `MergeKv` | `vector_mla.h` | 986 | MergeKv 主入口 |
| `GetRealS2Idx` | `vector_mla.h` | 821 | sparse block 索引映射为物理 KV 索引 |
| `GetKeyGmOffset` | `vector_mla.h` | 834 | 物理 KV 索引转换为 GM 偏移 |
| `CopyInKv` | `vector_mla.h` | 896 | 离散 KV 从 GM 搬运到 UB（两 block 合并路径） |
| `CopyInSingleKv` | `vector_mla.h` | 872 | 单 block 离散 KV 搬运（回退路径） |
| `CopyOutMrgeResult` | `vector_mla.h` | 960 | 合并后的连续 KV 写回 GM |
| `DataCopyPA` | `cube_mla.h` | 72 | PageAttention 场景的离散 KV 搬运 |
| `DataCopyGmNDToL1` | `cube_mla.h` | 47 | ND→NZ 格式转换 + 数据搬运 |
| `BlockAlign` | `common.h` | 59 | 数据对齐规则 |
| `Bmm2DataCopyOutTrans` | `vector_mla.h` | 1244 | 输出转置写回 |
| `Bmm2CastAndCopyOut` | `vector_mla.h` | 1259 | 输出类型转换 + 写回 |

## 9. 相关技术

- [[technique-sparse-attention-tiling]]：SparseFlashAttention 的 tiling 与分核策略，包含 MergeKv 所在的 V_TEMPLATE 说明。
- [[technique-flash-attention-pipeline]]：FlashAttention 的四阶段流水编排，MergeKv 位于 V0 阶段。
- [[technique-flash-attention-tiling]]：标准 FlashAttention 的 tiling 策略，不涉及稀疏索引。