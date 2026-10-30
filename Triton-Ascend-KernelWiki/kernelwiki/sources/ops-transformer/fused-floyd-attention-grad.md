---
id: ops-transformer-fused-floyd-attention-grad
title: CANN ops-transformer FusedFloydAttentionGrad source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/fused_floyd_attention_grad/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer FusedFloydAttentionGrad 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/fused_floyd_attention_grad/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。`fixture: true`。Floyd Attention Grad 是 Floyd Attention（Moonshot Kimi 长上下文双流注意力）的反向梯度算子。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/fused_floyd_attention_grad/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|---|---|---|
| `attention/fused_floyd_attention_grad/README.md` | `2870f402281f5e929a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3` | 算子总览、公式（6 个反向 matmul）、参数、约束 |
| `attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_def.cpp` | `129b803140567954f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | 算子定义（7 输入 5 输出/AICore config） |
| `attention/fused_floyd_attention_grad/op_kernel/fused_floyd_attention_grad.cpp` | `06d12b6c0de20625e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3` | kernel 入口（三段式 Pre→S1S2→Post + TilingKey 分发） |
| `attention/fused_floyd_attention_grad/op_kernel/fused_floyd_attention_grad_s1s2_bn2gs1s2.h` | `4abc8e9c487cc176f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3` | 6 个 matmul 反向核心实现 |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L17-52）

```
dV1 = P^T · dY
dV2 = P^T · dY
dQ = (dS·K1 + dS·K2) / √d
dK1 = (dS^T · Q) / √d
dK2 = (dS^T · Q) / √d
```

其中 `P = Softmax(scale*(Q·K1^T + Q·K2^T), mask)`，`dS = S * dY`。

### 3.2 三段式 kernel（来源: fused_floyd_attention_grad.cpp L41-76）

- `Pre`（`fused_floyd_attention_grad_pre.h`）：dS = softmax * dY
- `S1S2`（`fused_floyd_attention_grad_s1s2_bn2gs1s2.h`）：6 个 matmul 反向（dV1/dV2/dQ/dK1/dK2）
- `Post`（`fused_floyd_attention_grad_post.h`）：梯度搬运 + ND↔NZ 格式转换

### 3.3 架构分发逻辑（来源: fused_floyd_attention_grad_def.cpp L164-165）

`ascend910b`（A2 训练）/ `ascend910_93`（A3 训练）。无 950，无 310P。仅训练场景。

### 3.4 TilingKey 分发（来源: fused_floyd_attention_grad.cpp L91-120）

按 dtype 和 mask 分 4 个组合：BF16/FP16 × mask=0/mask=1。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/fused-floyd-attention-grad.md`
- `artifacts/fused-floyd-attention-grad/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要。SHA-256 值为示意（前 16 位），完整值需在固定提交下重算。tiling 参数数值、UB 字节预算、dS 精确公式需结合源码/硬件进一步验证。