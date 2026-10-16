---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: "4.1"
title: 禁止在 forward() 中使用 PyTorch 计算（Type-3 退化）
description: forward() 只做 shape/分派/host 预计算，所有数值计算必须在 @triton.jit kernel 内完成
tags: [no_pytorch_forward, type3_degradation, triton_only, host_preprocess]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-5 禁止 forward 中 PyTorch 计算

【优化维度】：移植｜正确性
【泛化层级】：Triton通用（迁移条件：所有 Triton-Ascend 算子；无失效边界）

<原则：validate_triton_impl.py Type-3 检查会 flag forward 中任何 torch 计算为退化。forward() 只负责 shape 计算、分派、host 预计算；所有数值计算必须在 @triton.jit kernel 内。也是 Triton-Ascend 算子基本要求。>

<骨架（占位名伪码）：
def forward(self, x, ...):
    x = x.contiguous()              # ✅ host 预处理
    shape = x.shape                 # ✅ shape 计算
    grid = (min(total, num_cores),) # ✅ 分派
    kernel[grid](x, out, ...)       # ✅ 数值计算在 kernel 内
    return out
    # 禁止：return torch.softmax(x)  # ❌ Type-3 退化
>

**坏实践（反例）**：forward 中调用 torch 计算
```python
def forward(self, x):
    return torch.softmax(x, dim=-1)  # ❌ 退化，Type-3 检查 flag
```
后果：validate_triton_impl.py Type-3 检查失败，算子判定退化。✅ 正解即所有数值计算移入 kernel。

**已知实例**：transformer-inference T3（MoeGatingTopKSoftmax 禁 torch.softmax/topk、MoeComputeExpertTokens 禁 torch.bincount/cumsum）、quantization Q6

<!-- okf:related:start -->

# 相关

- 实践案例: [FlashAttentionScore training-forward on Ascend NPU](../../ops/flash-attention-score.md) — 同属优化知识，主题相关
- 实践案例: [NsaSelectedAttention (NSA selected attention, training forward) on Ascend NPU](../../ops/nsa-selected-attention.md) — 同属优化知识，主题相关
- 相关主题: [多维输入必须在 host 侧扁平化为 2D](../memory-sram/host-flatten-to-2d.md) — 主题相关：共享 host_preprocess
- 相关主题: [输入进入 kernel 前必须 contiguous](../memory-sram/input-contiguous.md) — 主题相关：共享 host_preprocess
- 相关主题: [目标维度必须归一化到末维（permute->contiguous->处理->permute 回）](../memory-sram/normalize-dim-to-last.md) — 主题相关：共享 host_preprocess

<!-- okf:related:end -->
