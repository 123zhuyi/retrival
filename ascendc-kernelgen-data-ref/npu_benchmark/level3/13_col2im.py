import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# cv_02_col2im —— Col2ImV2 / fold（滑窗列重排回图像，融合版）
#
# 对应 ops-ref/ops-cv `col2_im_v2` AscendC kernel。
# kernel 算法：按输出像素 centric 滑窗，对每个输出位置累加所有覆盖它的
# 滑窗块贡献（h = h_col*stride - pad + h_k*dil），未被覆盖位置补 0，
# 单核完成列->图像重排 + 累加。
#
# 转换后的 PyTorch 直接调用 torch 原生融合算子 F.fold（即 col2im），
# 一次调用完成全部重排累加，不展开成逐位置小算子拼接。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Col2ImV2（融合版，forward 直接调用 F.fold）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, output_size, kernel_size, dilation, padding, stride):
        """
        Args:
            x (Tensor): [N, C*kH*kW, L]，滑窗列（im2col 输出）
            output_size (Tensor): int32 [2] = (H_out, W_out)
            kernel_size (Tensor): int32 [2] = (kH, kW)
            dilation (list[int]): (dilH, dilW)
            padding (list[int]): (padH, padW)
            stride (list[int]): (strideH, strideW)

        Returns:
            Tensor: [N, C, H_out, W_out]
        """
        out_size = tuple(int(v) for v in output_size.tolist())
        k_size = tuple(int(v) for v in kernel_size.tolist())
        return F.fold(x, out_size, k_size,
                      dilation=tuple(dilation),
                      padding=tuple(padding),
                      stride=tuple(stride))


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
            _random_tensor(specs["x"], 42 + case_index * 3),
            torch.tensor(specs["output_size"]["value"], dtype=torch.int32).npu(),
            torch.tensor(specs["kernel_size"]["value"], dtype=torch.int32).npu(),
            specs["dilation"]["value"],
            specs["padding"]["value"],
            specs["stride"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
