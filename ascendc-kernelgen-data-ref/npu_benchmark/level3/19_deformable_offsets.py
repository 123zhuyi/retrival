import json
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# cv_08_deformable_offsets —— IMGWarpOffsets（偏移量驱动的图像采样，融合版）
#
# 对应 ops-ref/ops-cv `image_warp_offsets` AscendC kernel（aclnnIMGWarpOffsets）。
# kernel 算法：对每个输出位置 (n, i, h, w)，把 offsets 张量中的元素视为输入
# 图像按行展开 [H, W, 3] 的线性索引，从 images[n] 连续拷出 3 个通道值组成
# warp_images[n, i, h, w, :]，单核一次完成偏移寻址 + 通道拷贝（等价于可变形
# 采样里“按偏移取点”这一步，整数索引、无插值）。
#
# 转换后的 PyTorch 用 torch.gather 一次完成全部偏移寻址：把 images 展平成
# [B, H*W*3]，对 offsets 加通道偏移 [0,1,2] 后 gather 出 3 通道，再 reshape
# 回 [B, 4, new_h, new_w, 3]（无 Python 逐位置循环）。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """IMGWarpOffsets（融合版，gather 一次算全部偏移采样）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, images, offsets):
        """
        Args:
            images (Tensor): [B, H, W, 3]（FLOAT16 / FLOAT32 / UINT8）
            offsets (Tensor): [B, 4, new_h, new_w]（FLOAT32 / INT32），
                              每个元素为 images[n] 展平 [H*W*3] 的线性索引

        Returns:
            Tensor: [B, 4, new_h, new_w, 3]，dtype = images dtype
        """
        b, h, w, ch = images.shape
        _, points, new_h, new_w = offsets.shape
        device = images.device

        images_flat = images.reshape(b, h * w * ch)             # [B, H*W*3]
        idx = offsets.to(torch.int64).reshape(b, -1)            # [B, 4*new_h*new_w]
        # 每个偏移指向一个线性位置，连续读 3 个通道
        warp_parts = [
            torch.gather(images_flat, 1, idx + c) for c in range(ch)
        ]
        warp_flat = torch.stack(warp_parts, dim=-1)             # [B, M, 3]
        return warp_flat.reshape(b, points, new_h, new_w, ch)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
}


def _random_images(spec, seed):
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


def _random_offsets(spec, seed):
    """随机生成合法偏移：值在 [0, H*W*3-3]，保证读取 3 通道不越界。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    max_index = spec["max_index"]
    if spec["dtype"] == "int32":
        return torch.randint(0, max_index, shape, generator=generator,
                             dtype=torch.int32).npu()
    tensor = torch.rand(shape, generator=generator) * max_index
    return tensor.float().npu()


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
        image_shape = specs["images"]["shape"]
        offsets_spec = specs["offsets"]
        offsets_spec["max_index"] = image_shape[1] * image_shape[2] * image_shape[3] - 2
        input_groups.append([
            _random_images(specs["images"], 42 + case_index * 3),
            _random_offsets(offsets_spec, 43 + case_index * 3),
        ])
    return input_groups


def get_init_inputs():
    return []
