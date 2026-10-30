import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cv_04_upsample_nearest3d —— UpsampleNearest3D（最近邻 3D 上采样，融合版）
#
# 对应 ops-ref/ops-cv `upsample_nearest3d` AscendC kernel
# （aclnnUpsampleNearest3D）。kernel 算法：按目标尺寸逐像素取最近邻源坐标，
# 单核完成坐标映射 + 采样，避免逐像素小算子拼接。
#
# 转换后的 PyTorch 直接调用 torch 原生融合算子 F.interpolate（nearest, 5D），
# 在 NPU 上分派到 CANN 的 UpsampleNearest3D AscendC kernel，一次调用完成。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """UpsampleNearest3D（融合版）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, output_size):
        """
        Args:
            input (Tensor): [N, C, D_in, H_in, W_in]
            output_size (list[int]): (D_out, H_out, W_out)

        Returns:
            Tensor: [N, C, D_out, H_out, W_out]
        """
        return F.interpolate(input, size=tuple(output_size), mode="nearest")


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
            specs["output_size"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
