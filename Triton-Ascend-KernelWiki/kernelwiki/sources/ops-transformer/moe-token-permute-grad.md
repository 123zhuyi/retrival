---
id: ops-transformer-moe-token-permute-grad
title: CANN ops-transformer MoeTokenPermuteGrad source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: moe/moe_token_permute_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenPermuteGrad 来源摘要

本页记录从 CANN ops-transformer 仓库 `moe/moe_token_permute_grad/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。MoE Token Permute Grad 是 `aclnnMoeTokenPermute` 的反向传播算子（unpermute）。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_token_permute_grad/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|---|---|---|
| `moe/moe_token_permute_grad/README.md` | `e05c7ea6e4bb53b89a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3` | 算子总览、公式（indexSelect+reshape+sum）、950 代理说明 |
| `moe/moe_token_permute_grad/op_host/moe_token_permute_grad_def.cpp` | `fd6e747da539d216f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | 算子定义（AICore config 910b/910_93，无 950） |
| `moe/moe_token_permute_grad/op_kernel/moe_token_permute_grad.cpp` | `0e823fe8e9865866e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3` | kernel 入口（调用 KernelMoeTokenUnpermute，TilingKey 按 dtype） |
| `moe/moe_token_permute_grad/op_kernel/moe_token_permute_grad.h` | `209bff4956abd9b6e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3` | unpermute 核心实现（indexSelect+reshape+sum） |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L18-29）

```
inputGrad = permutedOutputGrad.indexSelect(0, sortedIndices)
inputGrad = inputGrad.reshape(-1, numTopk, hiddenSize)
inputGrad = inputGrad.sum(dim=1)
```

### 3.2 kernel 实现（来源: moe_token_permute_grad.cpp L19-43）

调用 `KernelMoeTokenUnpermute<T, IdxT, OutT, PROBS=false>`（与 `moe_token_unpermute` 共享实现）。TilingKey 按 dtype：0=BF16 / 2=FP16 / 4=FP32。

### 3.3 950 代理关系（来源: README.md L84-90）

950 架构转调用 `aclnnMoeInitRoutingV2Grad`，参数映射：permutedOutputGrad->gradExpandedX、sortedIndices->expandedRowIdx、numTopk->topK、paddedMode->dropPadMode。

### 3.4 架构分发（来源: moe_token_permute_grad_def.cpp L46-47）

`ascend910b`/`ascend910_93`（A2/A3 训练），无 950 直接 config。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/moe-token-permute-grad.md`
- `artifacts/moe-token-permute-grad/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要。SHA-256 值为示意（前 16 位），完整值需在固定提交下重算。tiling 参数数值、950 代理正确性需结合源码/硬件进一步验证。