# nsa_compress_grad - 流水编排分析

来源：`op_kernel/nsa_compress_grad.h`（565 行核心实现）+ `op_host/nsa_compress_grad_tiling.cpp`（241 行 tiling）。NSA compress grad 为 NSA 压缩算子的反向，仅 A2 训练，单模板，无 Cube/Vector 交替流水（纯 Vector 计算）。

## 1. 核心计算公式

```
dw = dk_cmp · K^T    # weight 梯度：cmpGrad 与 input 点积沿 headDim reduce
dk = W^T · dk_cmp    # input 梯度：weight 广播后与 cmpGrad 逐元素乘
```

## 2. 单核流程

每个 AIV 处理若干 block（`totalBlockNum / coreNum` 外切），每 block 内按 head→row 循环：

```
for each block assigned to this core:
    CopyInCmpGrad(block)              # GM→UB：当前 block 的 cmpGrad
    for each head in block:
        for each row in blockSize:
            CopyInInput(row)          # GM→UB：对应 input KV
            CopyInWt(row)             # GM→UB：对应 weight
            ComputeWtGrad()           # V：Mul + BlockReduceSum + WholeReduceSum → dw
            CopyOutWtGrad()           # UB→GM：AtomicAdd weightGrad
            ComputeInputGrad()        # V：Cast + BroadCast + Mul → dk
            CopyOutInputGrad()        # UB→GM：AtomicAdd（overlap）或直接 store
```

## 3. 各阶段详情

| 阶段 | 函数 | 操作 | 数据流 |
|---|---|---|---|
| CopyInCmpGrad | `CopyInCmpGrad` L304-318 | GM→UB `DataCopyPad` + Cast(FP16→FP32) + Broadcast（nRowsOnce>1） | outputGradGm → cmpGradHf → cmpGrad |
| CopyInInput | `CopyInInput` L320-327 | GM→UB `DataCopyPad` input | inputGm → inputHf |
| CopyInWt | `CopyInWt` L393-399 | GM→UB `DataCopyPad` weight | weightGm → weightHf |
| ComputeWtGrad | `ComputeWtGrad` L330-349 | Cast(input→FP32) → Mul(cmpGrad) → BlockReduceSum（headDim/64 分段）→ WholeReduceSum | dw 在 tmpx |
| CopyOutWtGrad | `CopyOutWtGrad` L373-390 | `AtomicAdd` tmpx → wtGradAtomicWs | AtomicAdd 到 workspace |
| ComputeInputGrad | `ComputeInputGrad` L352-370 | Cast(weight→FP32) → BroadCast(沿 headDim) → Mul(cmpGrad) → CastByCondition | dk 在 inputGradHf/weightBrdc |
| CopyOutInputGrad | `CopyOutInputGrad` L402-416 | overlap: AtomicAdd → inputGradAtomicWs；非 overlap: DataCopyPad → inputGradGm | dk 到 output |

## 4. 主循环（`Process` L235-301）

```
PresetFlag()
for blcId in range(startBlcIdx_g, startBlcIdx_g + nBlockCurCore_):
    if blcId >= blcStartIdxCurBch + nBlcCurBch:     # 跨 batch 边界
        update curBIdx, blcStartIdxCurBch, batchOffset, seqLenCurBch, nBlcCurBch
    for headId in range(0, headNum_, nHeadOnce_):
        WaitFlag(EVENT_ID6)
        CopyInCmpGrad(cmpOffset, size)
        for rowId in range(0, blockSize_, nRowsOnce_):
            CopyInInput(inputOffset, size)
            CopyInWt(weightOffset, size)
            ComputeWtGrad(size)
            CopyOutWtGrad(weightOffset, size)
            ComputeInputGrad(size)
            CopyOutInputGrad(inputOffset, size)
        SetFlag(EVENT_ID6)
ClearPresetFlag()
```

## 5. 核间规约（`MoveResult` L419-437）

`SyncAll()` 后：
- `deterministicFlag_`：`DeterministicComputeSumWtGrad`（树形规约）
- 非 deterministic：`coreId_ == 0` 的核执行 `MoveWtGrad`（单一核收集 AtomicAdd 结果）
- `isOverLap_`：`MoveInputGrad`（ping-pong buffer 逐段搬运 inputGradAtomicWs → inputGradGm）

## 6. UB 内存布局

```
ubMaxSize_ - 24KB（UB_PRESERVE）
├── cmpGrad (FP32, maxHeadNumInUb_ × headDim)
├── inputFp32Ub_ (FP32, 同大小)
├── weightBrdc (FP32, 同大小)
├── tmpBuffer (uint8_t, 16KB)
├── tmpx (FP32, 8KB)
├── cmpGradHf (T, maxHeadNumInUb_ × headDim)
├── inputHf (T, 同大小)
├── weightHf (T, maxHeadNumInUb_)
├── weightFp32 (FP32, maxHeadNumInUb_)
├── wtGradHf / inputGradHf (T, 同区域复用)
├── transInputGradPing_/Pong_ (FP32 ping-pong)
├── halfTransInputGradPing_/Pong_ (T ping-pong)
└── transWtGrad_ / transDeterministicWtGrad_ (FP32)
```

## 7. Workspace

`workspace = 16MB（sysWorkspace）+ blockSize × headNum × aivNum × sizeof(float) + tSeqLen × headNum × headDim × sizeof(float) + 1MB`（`tiling.cpp` L143-150）。

其中：
- `wtGradAtomicWs_`：`blockSize × headNum × aivNum × sizeof(float)` — AtomicAdd 中间
- `inputGradAtomicWs_`（overlap 场景）：`tSeqLen × headNum × headDim × sizeof(float)` — overlap 累加中间

## 8. 阶段 → Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| dw = dk_cmp · K^T | `ComputeWtGrad`：Mul + BlockReduceSum + WholeReduceSum | `tl.dot(dk_cmp, K.T)` | 是（数学） |
| dk = W^T · dk_cmp | `ComputeInputGrad`：Cast + BroadCast + Mul | `tl.dot(W.T, dk_cmp)` | 是（数学） |
| AtomicAdd weightGrad | `SetAtomicAdd` + `DataCopyPad` | `tl.atomic_add` | 是（语义，精度 needs_verification） |
| overlap inputGrad | AtomicAdd 到 workspace + 后处理搬运 | `tl.atomic_add` 或 double buffer | 是（语义） |
| DataCopy GM→UB | MTE `DataCopyPad` + `SetFlag/WaitFlag` | `tl.load` | 否（机制） |
| Cast + Broadcast | AscendC Vector API | Triton `to()` + broadcast | 是（语义） |
| 核间同步 + 确定性规约 | `SyncAll` + `DeterministicComputeSumWtGrad` | Triton kernel launch barrier | 否（机制） |
| workspace ping-pong | 手动 `pingPongFlag` + `DataCopyPad` | Triton 编译器管理 | 否（机制） |