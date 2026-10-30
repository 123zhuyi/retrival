import json
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# cv_06_roi_align —— RoiAlign（ROI 对齐池化，融合版）
#
# 对应 ops-ref/ops-cv `roi_align` AscendC kernel（aclnnRoiAlign / aclnnRoiAlignV2，
# 后者对标 torchvision roi_align 原型）。
# kernel 算法：对每个 ROI，先按 spatial_scale 映射 ROI 坐标并（可选）平移 -0.5
# 对齐像素中心，再按 pooled 尺寸划分 bin，对每个 bin 内采样点做双线性插值后
# 取平均（mode=avg）或最大值（mode=max），单核一次完成坐标映射 + 越界处理 +
# 插值 + 池化四件事。
#
# 转换后的 PyTorch 不展开成逐像素小算子拼接，而是用广播一次算出全部 ROI x
# 全通道 x 全输出位置的采样坐标，再对每个采样点做一次向量化双线性插值，最后
# 归约出池化结果（无 Python 逐 ROI 循环）。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """RoiAlign（融合版，向量化一次算全部 ROI 的双线性采样 + 池化）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, boxes, pooled_height=3, pooled_width=3,
                spatial_scale=1.0, sampling_ratio=2, aligned=False, mode="avg"):
        """
        Args:
            input (Tensor): [B, C, H, W]（NCHW），FLOAT16 / FLOAT32
            boxes (Tensor): [K, 5]（image_id, x1, y1, x2, y2），坐标在输入空间尺度
            pooled_height (int): 池化输出高
            pooled_width (int): 池化输出宽
            spatial_scale (float): ROI 坐标 -> 特征图坐标的乘法尺度
            sampling_ratio (int): 每 bin 采样点数；<=0 时自适应为 ceil(bin)
            aligned (bool): True 时 ROI 坐标平移 -0.5（对齐像素中心）
            mode (str): "avg" 或 "max"

        Returns:
            Tensor: [K, C, pooled_height, pooled_width]
        """
        return _roi_align(input, boxes, (pooled_height, pooled_width),
                          spatial_scale, sampling_ratio, aligned, mode)


def _bilinear_sample(im, y, x):
    """向量化双线性插值。

    Args:
        im (Tensor): [K, C, H, W]，已按 batch index 排好
        y (Tensor): [K, ph, 1] 浮点行坐标
        x (Tensor): [K, 1, pw] 浮点列坐标

    Returns:
        Tensor: [K, C, ph, pw]，越界采样点贡献 0
    """
    k, c, h, w = im.shape
    y0 = torch.floor(y)
    x0 = torch.floor(x)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = y - y0
    wx = x - x0
    valid = (y0 >= 0) & (y1 <= h - 1) & (x0 >= 0) & (x1 <= w - 1)   # [K, ph, pw]
    y0c = y0.clamp(0, h - 1).long()
    y1c = y1.clamp(0, h - 1).long()
    x0c = x0.clamp(0, w - 1).long()
    x1c = x1.clamp(0, w - 1).long()
    kk = torch.arange(k, device=im.device)[:, None, None]            # [K, 1, 1]
    # im[kk, :, yy, xx] -> [K, ph, pw, C]
    v00 = im[kk, :, y0c, x0c]
    v01 = im[kk, :, y0c, x1c]
    v10 = im[kk, :, y1c, x0c]
    v11 = im[kk, :, y1c, x1c]
    w00 = ((1.0 - wy) * (1.0 - wx)).unsqueeze(-1)                     # [K, ph, pw, 1]
    w01 = ((1.0 - wy) * wx).unsqueeze(-1)
    w10 = (wy * (1.0 - wx)).unsqueeze(-1)
    w11 = (wy * wx).unsqueeze(-1)
    val = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    val = val * valid.unsqueeze(-1)
    return val.permute(0, 3, 1, 2)                                   # [K, C, ph, pw]


def _roi_align(input, boxes, pooled_size, spatial_scale, sampling_ratio,
               aligned, mode):
    b, c, h, w = input.shape
    k = boxes.shape[0]
    pooled_h, pooled_w = pooled_size

    batch_idx = boxes[:, 0].long()
    roi_start_w = boxes[:, 1] * spatial_scale
    roi_start_h = boxes[:, 2] * spatial_scale
    roi_end_w = boxes[:, 3] * spatial_scale
    roi_end_h = boxes[:, 4] * spatial_scale
    if not aligned:
        roi_start_w = roi_start_w - 0.5
        roi_start_h = roi_start_h - 0.5
        roi_end_w = roi_end_w - 0.5
        roi_end_h = roi_end_h - 0.5
    roi_w = (roi_end_w - roi_start_w).clamp(min=1.0)                 # [K]
    roi_h = (roi_end_h - roi_start_h).clamp(min=1.0)
    bin_w = roi_w / pooled_w
    bin_h = roi_h / pooled_h

    if sampling_ratio > 0:
        grid_h = torch.full((k,), float(sampling_ratio), device=input.device)
        grid_w = torch.full((k,), float(sampling_ratio), device=input.device)
    else:
        grid_h = torch.ceil(roi_h / pooled_h)                        # [K]
        grid_w = torch.ceil(roi_w / pooled_w)
    count = (grid_h * grid_w).clamp(min=1.0)

    im = input[batch_idx]                                            # [K, C, H, W]
    ph = torch.arange(pooled_h, device=input.device, dtype=torch.float32)
    pw = torch.arange(pooled_w, device=input.device, dtype=torch.float32)

    max_gh = int(grid_h.max().item())
    max_gw = int(grid_w.max().item())

    if mode == "max":
        out = torch.full((k, c, pooled_h, pooled_w), float("-inf"),
                         dtype=input.dtype, device=input.device)
    else:
        out = torch.zeros(k, c, pooled_h, pooled_w, dtype=input.dtype,
                          device=input.device)

    for iy in range(max_gh):
        for ix in range(max_gw):
            sample_ok = (iy < grid_h) & (ix < grid_w)                # [K]
            sample_ok = sample_ok[:, None, None, None]               # [K,1,1,1]
            y = (roi_start_h[:, None, None]
                 + ph[None, :, None] * bin_h[:, None, None]
                 + (iy + 0.5) * bin_h[:, None, None] / grid_h[:, None, None])
            x = (roi_start_w[:, None, None]
                 + pw[None, None, :] * bin_w[:, None, None]
                 + (ix + 0.5) * bin_w[:, None, None] / grid_w[:, None, None])
            val = _bilinear_sample(im, y, x)                         # [K, C, ph, pw]
            if mode == "max":
                val = torch.where(sample_ok, val,
                                  torch.full_like(val, float("-inf")))
                out = torch.maximum(out, val)
            else:
                out = out + val * sample_ok

    if mode == "max":
        out = torch.where(torch.isinf(out), torch.zeros_like(out), out)
    else:
        out = out / count[:, None, None, None]
    return out.to(input.dtype)


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


def _random_boxes(spec, seed):
    """随机生成合法 ROI：image_id 在 [0, B)，坐标满足 0<=x1<=x2<=W/ss 等。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])               # [K, 5]
    k = shape[0]
    b_max = spec.get("batch", 2)
    ss = spec.get("spatial_scale", 1.0)
    w_max = spec.get("w_max", 16.0) / ss
    h_max = spec.get("h_max", 16.0) / ss
    image_id = torch.randint(0, b_max, (k, 1), generator=generator).float()
    x1 = torch.rand(k, 1, generator=generator) * w_max * 0.6
    x2 = x1 + torch.rand(k, 1, generator=generator) * w_max * 0.6
    y1 = torch.rand(k, 1, generator=generator) * h_max * 0.6
    y2 = y1 + torch.rand(k, 1, generator=generator) * h_max * 0.6
    return torch.cat([image_id, x1, y1, x2, y2], dim=1).float().npu()


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
        boxes_spec = specs["boxes"]
        boxes_spec["batch"] = specs["input"]["shape"][0]
        boxes_spec["w_max"] = specs["input"]["shape"][3]
        boxes_spec["h_max"] = specs["input"]["shape"][2]
        boxes_spec["spatial_scale"] = specs["spatial_scale"]["value"]
        input_groups.append([
            _random_tensor(specs["input"], 42 + case_index * 3),
            _random_boxes(boxes_spec, 43 + case_index * 3),
            specs["pooled_height"]["value"],
            specs["pooled_width"]["value"],
            specs["spatial_scale"]["value"],
            specs["sampling_ratio"]["value"],
            specs["aligned"]["value"],
            specs["mode"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
