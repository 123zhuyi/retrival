import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cv_09_resize_nearest —— ResizeNearestNeighborV2（最近邻 resize，融合版）
#
# 对应 ops-ref/ops-cv `resize_nearest_neighbor_v2` AscendC kernel
# （aclnnUpsampleNearest2D / aclnnUpsampleNearest1D）。
# kernel 算法：按目标尺寸把输出位置映射回源坐标
#   y(H, W) = x(min(floor(H * scaleH), H-1), min(floor(W * scaleW), W-1))
#   scaleH = x_H / size[0]，scaleW = x_W / size[1]
# 单核完成坐标映射 + 最近邻采样，避免逐像素小算子拼接。
#
# 转换后的 PyTorch 直接调用 torch 原生融合算子 F.interpolate（mode="nearest"），
# 其默认（align_corners=False, half_pixel_centers=False）坐标映射与上面公式
# 完全一致；half_pixel_centers=True 时对应 mode="nearest-exact"。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """ResizeNearestNeighborV2（融合版）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, size, half_pixel_centers=False, align_corners=False):
        """
        Args:
            x (Tensor): [N, C, H, W]（NCHW），FLOAT16 / FLOAT32 / BFLOAT16
            size (list[int]): (out_h, out_w)
            half_pixel_centers (bool): True 时用 half-pixel 中心对齐
            align_corners (bool): True 时角像素中心对齐（nearest 下为边角情形）

        Returns:
            Tensor: [N, C, out_h, out_w]
        """
        out_h, out_w = int(size[0]), int(size[1])
        if half_pixel_centers:
            return F.interpolate(x, size=(out_h, out_w), mode="nearest-exact")
        if not align_corners:
            return F.interpolate(x, size=(out_h, out_w), mode="nearest")
        # align_corners=True 的 nearest：round(dst * (in-1)/(out-1))
        n, c, h, w = x.shape
        ph = torch.arange(out_h, device=x.device, dtype=torch.float32)
        pw = torch.arange(out_w, device=x.device, dtype=torch.float32)
        ih = torch.floor(ph * (h - 1) / (out_h - 1) + 0.5).long().clamp(0, h - 1)
        iw = torch.floor(pw * (w - 1) / (out_w - 1) + 0.5).long().clamp(0, w - 1)
        return x[:, :, ih[:, None], iw[None, :]]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    dtype = _DTYPE_MAP[spec["dtype"]]
    if torch.rand(1, generator=generator).item() < 0.5:
        mean = torch.rand(1, generator=generator).item() * 2.0 - 1.0
        std = torch.rand(1, generator=generator).item() * 0.9 + 0.1
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        tensor = tensor * std + mean
    else:
        tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
        tensor = tensor * 2.0 - 1.0
    return tensor.to(dtype=dtype).npu()


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
            _random_tensor(specs["x"], 42 + case_index * 3),
            specs["size"]["value"],
            specs["half_pixel_centers"]["value"],
            specs["align_corners"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
