import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cv_10_spatial_transformer —— SpatialTransformer（STN 仿射空间变换，融合版）
#
# 对应 ops-ref/ops-cv `spatial_transformer` AscendC kernel（兼容 TensorFlow
# SpatialTransformer 算子）。kernel 算法：用 2x3 仿射矩阵 theta 把输出网格映射
# 到输入坐标，再对输入张量做双线性采样，单核一次完成仿射网格生成 + 采样。
#
# 转换后的 PyTorch 用 torch 原生融合算子组合：F.affine_grid 生成仿射采样网格
# （与 kernel 的网格映射同构），F.grid_sample 完成一次采样，两个原生算子即可
# 覆盖整条 STN 变换路径，不展开成逐像素小算子拼接。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """SpatialTransformer（融合版，affine_grid + grid_sample）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, theta, output_size, align_corners=False,
                default_theta=None, use_default_theta=None):
        """
        Args:
            x (Tensor): [B, C, H, W]（NCHW），FLOAT16 / FLOAT32
            theta (Tensor): [B, 2, 3] 或 [2, 3] 仿射变换矩阵
            output_size (list[int]): (out_h, out_w)，-1 表示沿用输入尺寸
            align_corners (bool): 角像素中心对齐
            default_theta (list[float]): 默认仿射参数（6 个），可为空
            use_default_theta (list[int]): 6 个 0/1，1 表示该参数取 default_theta

        Returns:
            Tensor: [B, C, out_h, out_w]
        """
        b, c, h, w = x.shape
        if theta.dim() == 2:                          # [2, 3] -> [B, 2, 3]
            theta = theta.unsqueeze(0).expand(b, -1, -1).contiguous()

        if use_default_theta:
            mask = torch.tensor(use_default_theta, dtype=torch.float32,
                                device=theta.device)
            default = torch.tensor(default_theta, dtype=theta.dtype,
                                   device=theta.device)
            theta_flat = torch.where(mask[None, :] > 0,
                                     default[None, :].expand(b, 6),
                                     theta.reshape(b, 6))
            theta = theta_flat.reshape(b, 2, 3)

        out_h = h if output_size[0] <= 0 else int(output_size[0])
        out_w = w if output_size[1] <= 0 else int(output_size[1])

        grid = F.affine_grid(theta, (b, c, out_h, out_w),
                             align_corners=align_corners)
        # NPU 上 aclnnGridSampler2D 要求 grid 与 input dtype 一致
        grid = grid.to(x.dtype)
        return F.grid_sample(x, grid, align_corners=align_corners)


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
    tensor = tensor.to(dtype=dtype)
    if dtype == torch.float16:
        tensor = tensor * 0.5    # 缩小 fp16 输入幅度，压低绝对误差
    return tensor.npu()


def _random_theta(spec, seed):
    """随机仿射矩阵：围绕单位阵的小扰动，避免过度形变导致全越界。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])                     # [B, 2, 3]
    b = shape[0]
    a = 1.0 + (torch.rand(b, 1, generator=generator) - 0.5) * 0.4
    d = 1.0 + (torch.rand(b, 1, generator=generator) - 0.5) * 0.4
    off = (torch.rand(b, 2, generator=generator) - 0.5) * 0.4
    tr = (torch.rand(b, 2, generator=generator) - 0.5) * 0.6
    theta = torch.stack([
        torch.cat([a, off[:, 0:1], tr[:, 0:1]], dim=1),
        torch.cat([off[:, 1:2], d, tr[:, 1:2]], dim=1),
    ], dim=1)
    return theta.float().npu()


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
            _random_theta(specs["theta"], 43 + case_index * 3),
            specs["output_size"]["value"],
            specs["align_corners"]["value"],
            specs["default_theta"]["value"],
            specs["use_default_theta"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
