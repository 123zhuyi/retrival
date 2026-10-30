# nsa_compress_with_cache - tiling 与 dispatch 分析

来源：`attention/nsa_compress_with_cache/`（commit `d28d8f9...`）。

## A. 架构分发

`def.cpp` L62-67：`ascend910b` + `ascend910_93`（A2/A3，FP16/BF16）+ `kirinx90` + `kirin9030`（仅 FP16）。README L7-14：不支持 950/310P/训练系列。

## B. 核间切分（head × batch）

tiling 数据结构（`tiling.h` L24-50）：

- `coreStartBatchIdxList[50]` / `coreStartHeadIdxList[50]`：每核负责的 batch 范围 + head 起始。
- `headsNumPerCore`：每核每次处理 head 数。
- `coresNumPerCompress`：一个 token 压缩跨核数。
- `batchSize` / `headNum` / `headDim`：全局 shape。

kernel 入口 `InitBase`（kernel L43-48）：`blockIdx = GetBlockIdx()`，每核从 `coreStartBatchIdxList[blockIdx]` 到 `coreStartBatchIdxList[blockIdx+1]` 遍历 batch，每 batch 处理 `coreStartHeadIdxList[blockIdx]` 到 `coreStartHeadIdxList[blockIdx+1]` 的 head。

## C. 核内 tiling（压缩循环）

- `compressBlockSize`：压缩滑窗大小（16 倍数，≤64）。
- `compressStride`：压缩间隔（16 倍数，≤compressBlockSize）。
- `tokenNumPerTile` / `tokenRepeatCount`：核内循环一次压缩 token 数/循环次数。
- `bufferNum`：ping-pong buffer 数量。
- `alignReduceSize`：compressBlockSize 对齐到 2 的幂。
- `weightInitFlag`：weight 初始化方式。

## D. 压缩条件（kernel InitActSeqLen L66-78）

`seqLen >= compressBlockSize && (seqLen - compressBlockSize) % compressStride == 0`

## E. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockIdx` / coreStartBatchIdxList | `tl.program_id` / grid | 沿 head×batch 映射 |
| `headsNumPerCore` / `coresNumPerCompress` | `BLOCK_HEAD` constexpr | 静态 meta |
| compressBlockSize 滑动窗口 | `tl.load` 窗口 + `tl.sum` | 加权求和 |
| weight broadcast | `tl.load(weight)` × `tl.sum` | broadcast 乘 |
| slot_mapping scatter | `tl.store(cache + slot*stride)` | scatter 写回 |
| ping-pong buffer | `bufferNum` 循环 | 不迁移（编译器调度） |
| GetBlockIdx / coreStartXxxList | `tl.program_id` + grid | 映射 |