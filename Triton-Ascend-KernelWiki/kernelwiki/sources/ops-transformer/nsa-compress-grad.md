---
id: ops-transformer-nsa-compress-grad
title: CANN ops-transformer NsaCompressGrad source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/nsa_compress_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer NsaCompressGrad 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/nsa_compress_grad/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。NSA compress grad 是 NSA（Native Sparse Attention）压缩算子的反向计算，仅支持 Atlas A2 训练系列（arch32）。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/nsa_compress_grad/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|------|------|------|
| `attention/nsa_compress_grad/README.md` | `a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1` | 算子总览、公式、参数（仅 A2 训练） |
| `attention/nsa_compress_grad/docs/aclnnNsaCompressGrad.md` | `c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4` | aclnn API 定义 |
| `attention/nsa_compress_grad/op_host/nsa_compress_grad_def.cpp` | `d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3` | 算子定义（dtype/属性/AICore config） |
| `attention/nsa_compress_grad/op_host/nsa_compress_grad_tiling.cpp` | `e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5` | tiling 主逻辑（分核/workspace/校验） |
| `attention/nsa_compress_grad/op_host/nsa_compress_grad_tiling.h` | `f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6` | tiling 数据结构 |
| `attention/nsa_compress_grad/op_kernel/nsa_compress_grad.h` | `c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9` | kernel 核心实现（NsaCompressGradND 模板类） |
| `attention/nsa_compress_grad/op_kernel/nsa_compress_grad_base.h` | `d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0` | 基础工具（GetSeqLenTarBch/CompressBlkNum/CastByCondition） |
| `attention/nsa_compress_grad/op_kernel/nsa_compress_grad.cpp` | `b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8` | kernel 入口 |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L18-25）

- dw = dk_cmp · K^T（weight 梯度）
- dk = W^T · dk_cmp（input 梯度）

### 3.2 核心 kernel 实现（来源: nsa_compress_grad.h L330-370）

`ComputeWtGrad`（L330-349）：`Mul(inputFp32Ub, cmpGrad)` → `BlockReduceSum`（headDim 64 分段）→ `WholeReduceSum`（合段）→ AtomicAdd 到 workspace。

`ComputeInputGrad`（L352-370）：`Cast(weight→FP32)` → `BroadCast(沿 headDim)` → `Mul(weightBrdc, cmpGrad)` → `CastByCondition(→输出 dtype)` → AtomicAdd（overlap 场景）或直接 store。

### 3.3 主循环（来源: nsa_compress_grad.h L235-301）

`Process()`：遍历当前核的 block → head → row，每行复制 cmpGrad + input + weight，计算 weightGrad（AtomicAdd）、计算 inputGrad（AtomicAdd 或 store）。`MoveResult()` 规约并搬运最终结果。

### 3.4 分核逻辑（来源: nsa_compress_grad_tiling.cpp L87-136）

`DoOpTiling`：按 aivNum 沿 block 维外切。`calculateProcessInfo`（nsa_compress_grad.h L159-186）根据 cumsum actSeqLen 定位每个核的 startBatchIdx + batchOffset。

### 3.5 算子定义的 dtype/layout（来源: nsa_compress_grad_def.cpp）

- outputGrad/input/weight: FP16/BF16（ND）；inputGrad/weightGrad 输出同 dtype。
- 属性: compressBlockSize（必需/16 对齐）、compressStride（必需/16 对齐）、actSeqLenType（必需/0=cumsum）、layoutOptional（默认 TND）。
- AICore config: ascend910b（L72）、ascend910_93（L73），仅 A2 训练。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/nsa-compress-grad.md`
- `artifacts/nsa-compress-grad/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。SHA-256 值为示意（前 16 位），完整值需在固定提交下重算。tiling 参数数值、UB 字节预算、actSeqLenType=0 精确语义、AtomicAdd 多核精度边界需结合源码/硬件进一步验证。