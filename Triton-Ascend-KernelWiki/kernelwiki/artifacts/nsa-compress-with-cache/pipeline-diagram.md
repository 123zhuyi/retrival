# nsa_compress_with_cache - 流水编排分析

来源：`README.md` L17-29 + `op_kernel/nsa_compress_with_cache.cpp`。

## 1. 压缩流程

```
for each batch:
  if seqLen >= compressBlockSize && (seqLen - compressBlockSize) % compressStride == 0:
    compressIdx = (seqLen - compressBlockSize) / compressStride
    # 取后 compressBlockSize 个 token，与 weight 做加权求和
    input[tokens: compressIdx*stride : compressIdx*stride+compressBlockSize] × weight[:]
    # 写回 output_cache[slot_mapping[batch]]
```

## 2. 主循环

kernel 入口 `Process`（`kernel_mla.cpp`）：

| 阶段 | 操作 | 说明 |
|---|---|---|
| 1. InitBase | 读 tiling data，算每核 batch/head 范围，分配 ping-pong buffer | kernel L38-64 |
| 2. InitActSeqLen | 遍历 batch，检查 `(seqLen-compressBlockSize)%stride==0`，记录 compressBatchIdxList | L66-78 |
| 3. InitWeight | weight 从 GM 搬入 UB，Cast to FP32 | L80-91 |
| 4. InitOffset | 按 pingPongIdx 算当前 batchIdx/headIdx/seqLen/cacheOffset | L93-100 |
| 5. Compress loop | 滑动窗口内 token 加权求和 → scatter store | Process |

## 3. 关键计算（README L28-29）

```
outputCache[slot_mapping[i]] = Σ(input[compressIdx*stride : compressIdx*stride+compressBlockSize] × weight[:])
```

Triton 等价：`tl.sum(tl.load(input + offs_window) * tl.load(weight), axis=0)`

## 4. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| 压缩条件检查 | `actSeqLenGm.GetValue` + 条件判断 | `tl.load(seq_len)` + `if` | 是 |
| 滑动窗口加权求和 | DataCopy+Cast+Mul+ReduceSum | `tl.sum(tl.load(x)*weight, axis=0)` | 是 |
| slot_mapping gather+scatter | `slotMappingGm.GetValue` + store | `tl.store(cache + slot*stride)` | 是 |
| ping-pong buffer | inQueue/outQueue 循环 | 不迁移（编译器调度） | 否（CANN 专有） |
| GetBlockIdx 分核 | 硬件 API | `tl.program_id` | 映射 |