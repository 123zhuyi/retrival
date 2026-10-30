import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cv_03_resize_bilinear —— Resize（双线性，融合版）
#
# 对应 ops-ref/ops-cv `resize_bilinear_v2` AscendC kernel（aclnnResize）。
# kernel 算法：按 scale 映射目标像素到源坐标，双线性取四个近邻加权，
# 单核完成坐标映射 + 采样，避免逐像素小算子拼接。
#
# 转换后的 PyTorch 直接调用 torch 原生融合算子 F.interpolate（bilinear），
# 在 NPU 上分派到 CANN 的 Resize AscendC kernel，一次调用完成重采样。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Resize 双线性（融合版）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, scale_h, scale_w, align_corners=False):
        """
        Args:
            input (Tensor): [N, C, H_in, W_in]
            scale_h (float): 高方向缩放因子
            scale_w (float): 宽方向缩放因子
            align_corners (bool)

        Returns:
            Tensor: [N, C, H_in*scale_h, W_in*scale_w]
        """
        return F.interpolate(input, scale_factor=(scale_h, scale_w),
                             mode="bilinear", align_corners=align_corners)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
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
            _random_tensor(specs["input"], 42 + case_index * 3),
            specs["scale_h"]["value"],
            specs["scale_w"]["value"],
            specs["align_corners"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
