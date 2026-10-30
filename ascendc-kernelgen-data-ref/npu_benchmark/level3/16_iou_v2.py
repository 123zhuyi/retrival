import json
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# cv_05_iou_v2 —— IouV2（预测框与真实框 IoU/IOF，融合版）
#
# 对应 ops-ref/ops-cv `iou_v2` AscendC kernel（aclnnIouV2）。
# kernel 算法：对每对 (bbox, gtbox) 计算重叠面积 Area3，按 mode 计算
#   IoU = Area3 / (Area1 + Area2 - Area3)
#   IOF = Area3 / Area2
# 单核一次完成全部 box 对的重叠面积 + 归一化。
#
# 转换后的 PyTorch 用广播一次向量化算出全部 box 对的 IoU/IOF（aligned=false
# 时 bboxes[:,None,:] 与 gtboxes[None,:,:] 广播成 [N, M, 4]），无 Python 循环。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """IouV2（融合版，广播一次算全对 IoU/IOF）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, bboxes, gtboxes, mode="iou", eps=1e-5, aligned=False):
        """
        Args:
            bboxes (Tensor): [N, 4]（x1, y1, x2, y2）
            gtboxes (Tensor): [M, 4]（x3, y3, x4, y4）；aligned 时 [N, 4]
            mode (str): "iou" 或 "iof"
            eps (float): 分母保护项
            aligned (bool): True 时按行对应计算，输出 [N, 1]；否则 [N, M]

        Returns:
            Tensor: [N, M]（或 aligned 时 [N, 1]）
        """
        if aligned:
            b1 = bboxes
            b2 = gtboxes
        else:
            b1 = bboxes.unsqueeze(1)            # [N, 1, 4]
            b2 = gtboxes.unsqueeze(0)           # [1, M, 4]

        area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])   # bbox 面积
        area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])   # gtbox 面积

        inter_w = torch.clamp(torch.min(b1[..., 2], b2[..., 2]) - torch.max(b1[..., 0], b2[..., 0]), min=0.0)
        inter_h = torch.clamp(torch.min(b1[..., 3], b2[..., 3]) - torch.max(b1[..., 1], b2[..., 1]), min=0.0)
        inter = inter_w * inter_h

        if mode == "iou":
            denom = area1 + area2 - inter
        elif mode == "iof":
            denom = area2                      # 库上语义：前景面积 = gtbox
        else:
            raise ValueError(f"unsupported mode: {mode}")
        out = inter / (denom + eps)
        if aligned:
            out = out.unsqueeze(-1)              # kernel 约定 aligned 输出 [N, 1]
        return out


def _random_boxes(spec, seed, low=0.0, high=16.0):
    """随机生成合法 box：x1<x2, y1<y2。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])               # [N, 4]
    n = shape[0]
    x1 = torch.rand(n, 1, generator=generator) * (high - low) * 0.6 + low
    x2 = x1 + torch.rand(n, 1, generator=generator) * (high - low) * 0.4 + 0.5
    y1 = torch.rand(n, 1, generator=generator) * (high - low) * 0.6 + low
    y2 = y1 + torch.rand(n, 1, generator=generator) * (high - low) * 0.4 + 0.5
    return torch.cat([x1, y1, x2, y2], dim=1).float()


def _load_cases():
    json_path = os.path.splitext(__file__)[0] + ".json"
    with open(json_path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def _case_specs(case):
    return {item["name"]: item for item in case["inputs"]}


def get_input_groups():
    torch.manual_seed(42)
    input_groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = _case_specs(case)
        input_groups.append([
            _random_boxes(specs["bboxes"], 42 + case_index * 3),
            _random_boxes(specs["gtboxes"], 43 + case_index * 3),
            specs["mode"]["value"],
            specs["eps"]["value"],
            specs["aligned"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
