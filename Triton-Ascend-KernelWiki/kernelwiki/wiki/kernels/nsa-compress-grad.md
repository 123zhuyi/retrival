---
id: kernel-nsa-compress-grad
title: NsaCompressGrad (NSA compress backward) on Ascend NPU
type: kernel
tags: [nsa_compress_grad, nsa, attention, backward, compress, sparse_attention, elementwise, reduce]
techniques: [fp32_accumulation]
hardware_features: [unified_buffer, ai_core]
architectures: [Ascend910B1, Ascend910B2, Ascend910_93]
kernel_types: [attention]
operators: [nsa_compress_grad]
symptoms: [precision_mismatch, ub_overflow]
confidence: inferred
reproducibility: pseudocode
artifact_dir: artifacts/nsa-compress-grad
related: [kernel-flash-attention-score, technique-flash-attention-tiling]
sources: [ops-transformer-nsa-compress-grad]
performance_claims: []
---

# NsaCompressGrad (NSA compress backward)

CANN ops-transformer 中的 `NsaCompressGrad` 算子是 [NSA（Native Sparse Attention）](https://arxiv.org/abs/2502.11089) 压缩算子的**反向计算**。仅支持 Atlas A2 训练系列（arch32，ascend910b/910_93），仅 TND layout。核心计算为两个 matmul 反向（`dw = dk_cmp · K^T`，`dk = W^T · dk_cmp`），纯 Vector 计算，无 Cube。

## 1. 输入输出

### 输入（按 `nsa_compress_grad_def.cpp` L22-49）

| 输入 | 必选/可选 | 含义 | dtype |
|------|-----------|------|-------|
| outputGrad | 必选 | 正向算子输出的反向梯度（dk_cmp），shape [T, N, D] | BF16/FP16 |
| input | 必选 | 待压缩张量（K），shape [T, N, D] | BF16/FP16 |
| weight | 必选 | 压缩权重（W），shape [compressBlockSize, N]，与 input broadcast | BF16/FP16 |
| actSeqLen | 可选 | 每个 batch 的 S 大小（cumsum 格式，actSeqLenType=0） | INT64 |

### 输出

| 输出 | 含义 | dtype |
|------|------|-------|
| inputGrad | input 的梯度（dk），shape 同 input | BF16/FP16 |
| weightGrad | weight 的梯度（dw），shape 同 weight | BF16/FP16 |

### 属性

| 属性 | 含义 | 说明 |
|------|------|------|
| compressBlockSize | 压缩滑窗大小 | 必需，16 对齐 |
| compressStride | 两次压缩滑窗间隔大小 | 必需，16 对齐，compressBlockSize > compressStride |
| actSeqLenType | 序列长度类型 | 必需，0=cumsum 结果（当前仅支持 0） |
| layoutOptional | 输入排布格式 | 可选，默认 TND |

### 维度约束

- compressBlockSize 和 compressStride 必须是 16 的整数倍，且 compressBlockSize > compressStride（`tiling.cpp` L220-233）。
- headDim 必须是 16 的整数倍（`tiling.cpp` L231-233）。

## 2. 核心计算公式

```
dw = dk_cmp · K^T    # weight 梯度：沿 headDim 的 matmul reduce
dk = W^T · dk_cmp    # input 梯度：weight 沿 headDim 广播后逐元素乘
```

CANN kernel 实现：
- `ComputeWtGrad`（`nsa_compress_grad.h` L330-349）：`Mul(inputFp32Ub, cmpGrad)` → `BlockReduceSum`（headDim 64 分段）→ `WholeReduceSum`（合段）→ `AtomicAdd` 到 workspace。
- `ComputeInputGrad`（`nsa_compress_grad.h` L352-370）：`Cast(weight→FP32)` → `BroadCast（沿 headDim）` → `Mul(weightBrdc, cmpGrad)` → `CastByCondition（→输出 dtype）`。

## 3. 计算流程

### 3.1 分核（按 aivNum 沿 block 维外切）

```
totalBlockNum = outputGrad.shape[0]（压缩后的 block 数）
nBlockPerCore = totalBlockNum / aivNum
剩余块分配给前 nRemainder 个核
```

### 3.2 单核主循环

```
for each block assigned to this core:
    CopyInCmpGrad(block)              # GM→UB：当前 block 的 dk_cmp
    for each head in block:
        for each row in blockSize:
            CopyInInput(row)          # GM→UB：对应 input KV
            CopyInWt(row)             # GM→UB：对应 weight
            ComputeWtGrad()           # Mul + reduce → dw
            CopyOutWtGrad()           # AtomicAdd → workspace
            ComputeInputGrad()        # Cast + BroadCast + Mul → dk
            CopyOutInputGrad()        # AtomicAdd（overlap）或 store
```

### 3.3 核间规约

`MoveResult()`（`nsa_compress_grad.h` L419-437）：`SyncAll()` 后按 `deterministicFlag_` 分支：
- deterministic：`DeterministicComputeSumWtGrad`（树形规约）。
- 非 deterministic：`coreId_==0` 的核收集 AtomicAdd 结果。
- `isOverLap_`：`MoveInputGrad`（ping-pong buffer 搬运 workspace → output）。

## 4. 架构与产品支持

| 产品 | 架构 | config（def.cpp） | kernel 入口 | tiling |
|------|------|-------------------|-------------|--------|
| Atlas A2 训练 | arch32（910B） | `ascend910b` L72 | `nsa_compress_grad.cpp` | `tiling.cpp` |
| Atlas A3 训练 | ascend910_93 | `ascend910_93` L73 | 同上 | 同上 |

> 不支持 Ascend 950、310P、推理加速卡、A2 推理。TilingKey 恒为 0（单模板）。

## 5. 工作机制

### 5.1 compress 滑窗

正向 NSA compress 用滑动窗口对 input 压缩：每个窗口大小 `compressBlockSize`，步长 `compressStride`，输出 `blockNum` 个压缩后的 token。反向计算 `dk` 和 `dw`。

### 5.2 AtomicAdd 规约

多核可能写 `weightGrad` 同一位置（不同 block 的 weight 梯度累加），CANN 用 `SetAtomicAdd<float>` + `DataCopyPad` 实现原子加。Triton 可迁移为 `tl.atomic_add`。

### 5.3 overlap 处理

`compressBlockSize > compressStride` 时，相邻窗口有 overlap，input 的梯度需要累加。CANN 用 `isOverLap_` 标志控制 AtomicAdd 写 workspace 后再搬运。

### 5.4 变长 TND

`actSeqLenType=0` 表示 cumsum 格式，`GetSeqLenTarBch`（`nsa_compress_grad_base.h`）解析每个 batch 的序列长度。

## 6. Agent actions

1. 确认目标架构为 A2 训练（ascend910b/910_93），仅 TND layout。
2. 确认 `compressBlockSize` 与 `compressStride` 16 对齐且 `compressBlockSize > compressStride`。
3. 分核：`grid = (totalBlockNum,)` 沿 block 维外切。
4. 核心计算：`tl.dot(dk_cmp, K.T)`（weight 梯度）+ `tl.dot(W.T, dk_cmp)`（input 梯度）。
5. 多核写 `weightGrad` 同位置 → `tl.atomic_add`；overlap 场景 `inputGrad` 也需 atomic_add。
6. 变长支持：`actSeqLen` cumsum → per-batch offset + mask。
7. Triton 实现参考 `artifacts/nsa-compress-grad/nsa-compress-grad-triton-skeleton.py`（pseudocode）。

## 7. 复现性声明

本页结论基于源码静态分析（confidence: inferred）。tiling 参数数值、UB 字节预算、actSeqLenType=0 精确语义、AtomicAdd 多核精度边界、确定性规约算法差异需结合源码/硬件进一步验证。Triton skeleton 标记 `reproducibility: pseudocode`，不可直接编译运行。来源见 [[ops-transformer-nsa-compress-grad]]。