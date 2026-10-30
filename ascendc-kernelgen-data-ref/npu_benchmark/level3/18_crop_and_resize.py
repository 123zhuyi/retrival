import json
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# cv_07_crop_and_resize —— CropAndResize（按归一化框裁剪 + resize，融合版）
#
# 对应 ops-ref/ops-cv `crop_and_resize` AscendC kernel（aclnnCropAndResize，
# 兼容 tf.image.crop_and_resize）。
# kernel 算法：对每个 box（归一化 [y1, x1, y2, x2]），按 crop_size 把输出位置
# 映射回输入图像坐标（bilinear/nearest/bilinear_v2 三种插值），单核一次完成
# 裁剪窗口坐标映射 + 插值采样 + 越界外插三件事。
#
# 转换后的 PyTorch 不展开成逐 box 小算子拼接，而是用广播一次算出全部 box x
# 全部输出位置 x 全部通道的采样坐标与权重，再向量化 gather 四个插值角点，
# 最后用 where 完成越界外插（无 Python 逐 box 循环）。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """CropAndResize（融合版，向量化一次算全部 box 的裁剪 + resize）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, boxes, box_index, crop_size,
                extrapolation_value=0.0, method="bilinear"):
        """
        Args:
            x (Tensor): [N, H, W, C]（NHWC），FLOAT16 / FLOAT32
            boxes (Tensor): [num_boxes, 4]（归一化 [y1, x1, y2, x2]），FLOAT32
            box_index (Tensor): [num_boxes] INT32，每 box 对应的 batch 索引
            crop_size (list[int]): (crop_height, crop_width)
            extrapolation_value (float): 越界外插值
            method (str): "bilinear" / "nearest" / "bilinear_v2"

        Returns:
            Tensor: [num_boxes, crop_height, crop_width, C]，dtype = boxes dtype
        """
        return _crop_and_resize(x, boxes, box_index, crop_size,
                                extrapolation_value, method)


def _crop_and_resize(x, boxes, box_index, crop_size,
                     extrapolation_value, method):
    n, h, w, c = x.shape
    num_boxes = boxes.shape[0]
    crop_h = int(crop_size[0])
    crop_w = int(crop_size[1])
    device = x.device

    xb = x[box_index]                                   # [num_boxes, H, W, C]
    xb = xb.to(torch.float32)                           # 中间一律 float32 保精度
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    py = torch.arange(crop_h, device=device, dtype=torch.float32)
    px = torch.arange(crop_w, device=device, dtype=torch.float32)

    if method in ("bilinear", "nearest"):
        if crop_h > 1:
            height_scale = (y2 - y1) * (h - 1) / (crop_h - 1)
            in_y = (y1 * (h - 1))[:, None] + py[None, :] * height_scale[:, None]
        else:
            in_y = 0.5 * (y1 + y2) * (h - 1)
            in_y = in_y[:, None].expand(num_boxes, crop_h)
        if crop_w > 1:
            width_scale = (x2 - x1) * (w - 1) / (crop_w - 1)
            in_x = (x1 * (w - 1))[:, None] + px[None, :] * width_scale[:, None]
        else:
            in_x = 0.5 * (x1 + x2) * (w - 1)
            in_x = in_x[:, None].expand(num_boxes, crop_w)

        valid = ((in_y[:, :, None] >= 0) & (in_y[:, :, None] <= h - 1)
                 & (in_x[:, None, :] >= 0) & (in_x[:, None, :] <= w - 1))

        if method == "bilinear":
            top = torch.floor(in_y)
            bottom = torch.ceil(in_y)
            left = torch.floor(in_x)
            right = torch.ceil(in_x)
            y_ratio = in_y - top
            x_ratio = in_x - left
            top_c = top.clamp(0, h - 1).long()[:, :, None]
            bottom_c = bottom.clamp(0, h - 1).long()[:, :, None]
            left_c = left.clamp(0, w - 1).long()[:, None, :]
            right_c = right.clamp(0, w - 1).long()[:, None, :]
            kk = torch.arange(num_boxes, device=device)[:, None, None]
            v_tl = xb[kk, top_c, left_c]
            v_tr = xb[kk, top_c, right_c]
            v_bl = xb[kk, bottom_c, left_c]
            v_br = xb[kk, bottom_c, right_c]
            wy = y_ratio[:, :, None]
            wx = x_ratio[:, None, :]
            val = (((1.0 - wy) * (1.0 - wx)).unsqueeze(-1) * v_tl
                   + ((1.0 - wy) * wx).unsqueeze(-1) * v_tr
                   + (wy * (1.0 - wx)).unsqueeze(-1) * v_bl
                   + (wy * wx).unsqueeze(-1) * v_br)
        else:  # nearest：四舍五入 = floor(v + 0.5)
            iy = torch.floor(in_y + 0.5).clamp(0, h - 1).long()[:, :, None]
            ix = torch.floor(in_x + 0.5).clamp(0, w - 1).long()[:, None, :]
            kk = torch.arange(num_boxes, device=device)[:, None, None]
            val = xb[kk, iy, ix]                        # [num_boxes, crop_h, crop_w, C]

        out = torch.where(valid.unsqueeze(-1), val,
                          torch.full_like(val, extrapolation_value))
    elif method == "bilinear_v2":
        y1o = torch.floor(y1 * h).long()
        x1o = torch.floor(x1 * w).long()
        y2o = torch.floor(y2 * h).long()
        x2o = torch.floor(x2 * w).long()
        ch = (y2o - y1o + 1).clamp(min=1).float()       # [num_boxes]
        cw = (x2o - x1o + 1).clamp(min=1).float()
        r_y = (py[None, :] + 0.5) * ch[:, None] / crop_h - 0.5
        r_x = (px[None, :] + 0.5) * cw[:, None] / crop_w - 0.5
        zero_h = torch.zeros_like(ch[:, None] - 1)
        zero_w = torch.zeros_like(cw[:, None] - 1)
        lower_y = torch.floor(r_y).clamp(zero_h, ch[:, None] - 1).long()
        upper_y = torch.ceil(r_y).clamp(zero_h, ch[:, None] - 1).long()
        lower_x = torch.floor(r_x).clamp(zero_w, cw[:, None] - 1).long()
        upper_x = torch.ceil(r_x).clamp(zero_w, cw[:, None] - 1).long()
        y_ratio = (r_y - lower_y.float())[:, :, None]
        x_ratio = (r_x - lower_x.float())[:, None, :]
        gy0 = (y1o[:, None] + lower_y).clamp(0, h - 1)[:, :, None]
        gy1 = (y1o[:, None] + upper_y).clamp(0, h - 1)[:, :, None]
        gx0 = (x1o[:, None] + lower_x).clamp(0, w - 1)[:, None, :]
        gx1 = (x1o[:, None] + upper_x).clamp(0, w - 1)[:, None, :]
        kk = torch.arange(num_boxes, device=device)[:, None, None]
        v_tl = xb[kk, gy0, gx0]
        v_tr = xb[kk, gy0, gx1]
        v_bl = xb[kk, gy1, gx0]
        v_br = xb[kk, gy1, gx1]
        out = (((1.0 - y_ratio) * (1.0 - x_ratio)).unsqueeze(-1) * v_tl
               + ((1.0 - y_ratio) * x_ratio).unsqueeze(-1) * v_tr
               + (y_ratio * (1.0 - x_ratio)).unsqueeze(-1) * v_bl
               + (y_ratio * x_ratio).unsqueeze(-1) * v_br)
    else:
        raise ValueError(f"unsupported method: {method}")

    return out.to(boxes.dtype)


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
    """随机生成归一化合法框：y1<y2, x1<x2 且落在 [0,1]。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    n = shape[0]
    y1 = torch.rand(n, 1, generator=generator) * 0.7
    y2 = y1 + torch.rand(n, 1, generator=generator) * 0.3
    x1 = torch.rand(n, 1, generator=generator) * 0.7
    x2 = x1 + torch.rand(n, 1, generator=generator) * 0.3
    return torch.cat([y1, x1, y2, x2], dim=1).float().npu()


def _random_box_index(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    n = shape[0]
    batch = spec.get("batch", 2)
    return torch.randint(0, batch, (n,), generator=generator,
                         dtype=torch.int32).npu()


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
        box_index_spec = specs["box_index"]
        box_index_spec["batch"] = specs["x"]["shape"][0]
        input_groups.append([
            _random_tensor(specs["x"], 42 + case_index * 3),
            _random_boxes(specs["boxes"], 43 + case_index * 3),
            _random_box_index(box_index_spec, 44 + case_index * 3),
            specs["crop_size"]["value"],
            specs["extrapolation_value"]["value"],
            specs["method"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
