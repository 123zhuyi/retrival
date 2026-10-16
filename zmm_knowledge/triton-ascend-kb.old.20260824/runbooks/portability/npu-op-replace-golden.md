---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: runbooks
category: portability
dimension: "4.1"
title: NPU 原生算子与标准算法不兼容时必须替换 Golden
description: NPU 原生算子（npu_nms/npu_iou）内部不透明且与标准算法有系统性差异，必须用纯 PyTorch 数学公式作 golden
tags: [npu_native_op, replace_golden, nms, iou, opacity]
confidence: verified
status: verified
created_at: "2026-08-05T00:00:00Z"
updated_at: "2026-08-05T00:00:00Z"
---

## OPT-6 NPU 原生算子替换 Golden

【优化维度】：移植｜正确性
【泛化层级】：条件性（迁移条件：task 文件含 NPU 原生算子且与标准算法有数值差异；失效边界：NPU 算子与标准算法一致）

<原则：NPU 原生算子（npu_nms_v4/npu_iou）内部实现不透明（fp16 中间计算、不同 eps、变长输出），且与标准算法存在系统性数值差异（约 1e-3，NPU 结果总是略小于标准算法）。禁止逆向工程 NPU 算子内部规则；必须以纯 PyTorch 数学公式实现作为 golden，Triton 实现按标准算法编写天然匹配。>

<骨架（占位名伪码）：
# task 文件含多个 Model 类时，选纯 PyTorch 数学公式实现作 golden
class Model(nn.Module):
    def forward(self, boxes, scores, ...):
        # 纯 PyTorch 数学公式（非 torch_npu.npu_nms_v4）
        order = scores.argsort(descending=True)
        boxes = boxes[order]
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            # 标准 IoU 计算
            ious = compute_iou(boxes[i], boxes[order[1:]])
            order = order[1:][ious < threshold]
        return torch.as_tensor(keep)
>

**坏实践（反例）**：逆向工程 NPU 原生算子
```python
# 试图匹配 npu_nms_v4 的内部排序/抑制规则（fp16 中间、eps 差异）
```
后果：系统性差异约 1e-3 无法消除，verify 持续失败。✅ 正解即替换为标准算法 golden。

**已知实例**：object-detection D3（npu_nms_v4/npu_iou 替换为纯 PyTorch 数学公式）
