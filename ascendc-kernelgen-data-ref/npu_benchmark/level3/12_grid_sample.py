import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cv_01_grid_sample —— GridSample 2D/3D 采样（融合版）
#
# 对应 ops-ref/ops-cv `grid_sample` AscendC kernel（aclnnGridSampler2D/3D）。
# kernel 算法：对每个输出位置，由 grid 归一化坐标 -> 反算输入坐标 -> 按
# interpolationMode(bilinear/nearest/bicubic) 与 paddingMode(zeros/border/
# reflection) 插值采样，单核完成坐标映射 + 越界处理 + 插值三件事。
#
# 转换后的 PyTorch 直接调用 torch 原生融合算子 F.grid_sample：在 NPU 上它
# 分派到 CANN 的 GridSampler AscendC kernel，与 ops-ref 实现同源，一次调用
# 完成全部采样逻辑（不展开成逐像素小算子拼接）。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """GridSample 2D/3D（融合版，forward 直接调用原生融合算子）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, grid, mode="bilinear", padding_mode="zeros",
                align_corners=False):
        """
        Args:
            input (Tensor): [N, C, H_in, W_in] 或 [N, C, D_in, H_in, W_in]
            grid (Tensor): [N, H_out, W_out, 2] 或 [N, D_out, H_out, W_out, 3]
            mode (str): bilinear / nearest / bicubic
            padding_mode (str): zeros / border / reflection
            align_corners (bool)

        Returns:
            Tensor: [N, C, H_out, W_out]（或 5D）
        """
        return F.grid_sample(input, grid, mode=mode,
                             padding_mode=padding_mode,
                             align_corners=align_corners)


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


def _random_grid(spec, seed):
    """grid 坐标归一化到 [-1, 1]，与 kernel 约定一致。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    dtype = _DTYPE_MAP[spec["dtype"]]
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
        grid_name = "grid" if "grid" in specs else "grid5d"
        input_groups.append([
            _random_tensor(specs["input"], 42 + case_index * 3),
            _random_grid(specs[grid_name], 43 + case_index * 3),
            specs["mode"]["value"],
            specs["padding_mode"]["value"],
            specs["align_corners"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
