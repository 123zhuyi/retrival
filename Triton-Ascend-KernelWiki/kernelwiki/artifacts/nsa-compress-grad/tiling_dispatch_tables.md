# nsa_compress_grad - tiling 与 dispatch 分析

来源：`attention/nsa_compress_grad/`（commit `d28d8f9…`）。NSA compress grad 为 NSA 压缩算子的反向，仅 A2 训练（arch32），TilingKey 恒为 0（单模板）。tiling 参数数值 `needs_verification`。

## A. 架构分发

`def.cpp` L72-73 注册 AICore config：

| 架构 | SocVersion | config | kernel 入口 | tiling |
|---|---|---|---|---|
| Atlas A2 训练 | `ascend910b` L72 | `nsa_compress_grad` | `op_kernel/nsa_compress_grad.cpp` | `tiling.cpp` |
| Atlas A3 训练 | `ascend910_93` L73 | 同上（arch32） | 同上 | 同上 |

> 仅支持 A2 训练系列（README L8），无 arch35/arch20。

## B. TilingKey

`GetTilingKey()`（`tiling.cpp` L152-155）恒返回 0。单模板（`nsa_compress_grad.h` 中的 `NsaCompressGradND<T>`）。

## C. 核间切分

| 项 | 结论 | 来源 |
|---|---|---|
| 逻辑轴 | block 维（totalBlockNum） | `tiling.cpp` L94 |
| 切分方式 | block 轴外切：`nBlockPerCore = totalBlockNum / coreNum`，剩余块分配给前 nRemainder 个核 | `nsa_compress_grad.h` L160-167（`calculateProcessInfo`） |
| 分核数 | `SetBlockDim(aivNum)` | `tiling.cpp` L168 |
| per-batch 解析 | `calculateProcessInfo` 根据 cumsum 定位当前核的 startBatchIdx + batchOffset | `nsa_compress_grad.h` L159-186 |

## D. 核内处理

| 单元 | 说明 | 来源 |
|---|---|---|
| block 遍历 | 当前核从 `startBlcIdx_g` 开始处理 `nBlockCurCore_` 个 block | `nsa_compress_grad.h` L257-299 |
| head 遍历 | 每块按 `nHeadOnce_` 步长切分 head 维 | L271-298 |
| row 遍历 | 每 head 按 `nRowsOnce_` 步长沿 blockSize 行遍历 | L279-297 |
| 每 row 操作 | CopyInCmpGrad → CopyInInput → CopyInWt → ComputeWtGrad → CopyOutWtGrad → ComputeInputGrad → CopyOutInputGrad | L280-295 |
| UB 使用 | `maxHeadNumInUb_ = (ubCalcSize / headDim / sizeof(float) / 3 / 16) * 16` | L71 |

## E. 计算公式

```
dw = dk_cmp · K^T    # weight 梯度：cmpGrad × input 的 matmul reduce
dk = W^T · dk_cmp    # input 梯度：weight 广播后 × cmpGrad
```

CANN kernel 实现：
- `ComputeWtGrad`（L330-349）：`Mul(inputFp32Ub, cmpGrad)` → `BlockReduceSum`（沿 headDim 64 分段）→ `WholeReduceSum`（合段）
- `ComputeInputGrad`（L352-370）：`Cast(weight→FP32)` → `BroadCast(沿 headDim)` → `Mul(weightBrdc, cmpGrad)` → `CastByCondition(→output dtype)`

## F. Triton-Ascend 抽象映射

| CANN 概念 | Triton-Ascend 抽象 | 说明 |
|---|---|---|
| `blockNum`/core id | `grid = (totalBlockNum,)` + `tl.program_id(0)` | block 维外切 |
| blockSize/blockStride | `BLOCK_SIZE: tl.constexpr` / `BLOCK_STRIDE: tl.constexpr` | 16 对齐 |
| ComputeWtGrad（Mul+ReduceSum） | `tl.dot(dk_cmp, K.T)` | matmul 反向语义 |
| ComputeInputGrad（BroadCast+Mul） | `tl.dot(W.T, dk_cmp)` | matmul 反向语义 |
| AtomicAdd weightGrad | `tl.atomic_add` | 多核写同位置 |
| overlap inputGrad | `tl.atomic_add` 或 double buffer | blockSize > blockStride |
| actSeqLen cumsum | per-batch offset + mask | 变长 |
| TilingKey=0 | 单 `@triton.jit` kernel | 无分支 |
| MTE DataCopyPad / SetFlag/WaitFlag | 不迁移（Triton load/store） | CANN 专有 |