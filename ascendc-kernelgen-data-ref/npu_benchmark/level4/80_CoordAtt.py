import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


def _conv1x1_ref(x, weight, bias=None):
    """1x1 conv 的显式 matmul 参考实现，使用 fp32 累加再回 cast，减弱 NPU/Triton 累加序差异。"""
    orig_dtype = x.dtype
    x = x.float()
    weight = weight.squeeze(-1).squeeze(-1).float()
    out = torch.einsum('oc,bchw->bohw', weight, x)
    if bias is not None:
        out = out + bias.float().view(1, -1, 1, 1)
    return out.to(orig_dtype)


class Model(nn.Module):
    """
    Coordinate Attention.

    模块语义：
        x_h = adaptive_avg_pool2d(x, (None, 1))
        x_w = adaptive_avg_pool2d(x, (1, None)).permute(0,1,3,2)
        y = Conv1x1(cat([x_h, x_w])) + BN + h_swish
        (x_h, x_w) = split(y, [H, W])
        a_h = sigmoid(Conv1x1(x_h))
        a_w = sigmoid(Conv1x1(x_w))
        out = x * a_w * a_h
    Conv/BN 都在 Coordinate Attention 公开定义内，权重传入 forward。
    BN 使用 batch statistics（与首次调用一致）。

    输入约束（由 case 保证）：
      x: (B, C, H, W)
      inp/oup: 保留接口兼容，实际使用 x.shape[1]
      reduction: bottleneck 缩减比
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, inp, oup, reduction,
                conv1_w, conv1_b, bn_w, bn_b,
                conv_h_w, conv_h_b, conv_w_w, conv_w_b):
        identity = x
        b, c, h, w = x.shape
        mip = max(8, c // reduction)

        x_h = F.adaptive_avg_pool2d(x, (None, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, None)).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = _conv1x1_ref(y, conv1_w, conv1_b)
        y = F.batch_norm(y, torch.zeros_like(bn_w), torch.ones_like(bn_w),
                         bn_w, bn_b, training=True, momentum=0.1, eps=1e-5)
        y = y * (F.relu6(y + 3) / 6)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(_conv1x1_ref(x_h, conv_h_w, conv_h_b))
        a_w = torch.sigmoid(_conv1x1_ref(x_w, conv_w_w, conv_w_b))

        return identity * a_w * a_h


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(shape, dtype, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if torch.rand(1, generator=generator).item() < 0.5:
        mu = float(torch.empty(1).uniform_(-2.0, 2.0).item())
        sigma = float(torch.empty(1).uniform_(0.1, 1.0).item())
        if dtype is torch.bfloat16:
            return torch.normal(mu, sigma, shape, generator=generator, dtype=torch.float32).to(dtype)
        return torch.normal(mu, sigma, shape, generator=generator, dtype=dtype)
    else:
        t = torch.empty(shape, dtype=dtype)
        t.uniform_(-2.0, 2.0, generator=generator)
        return t


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "80_CoordAtt.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = _DTYPE_MAP[x_info["dtype"]]
        x = _random_tensor(x_info["shape"], dtype, 42 + case_index)
        c = x.shape[1]

        def attr(name, default=None):
            info = next((i for i in inputs if i["name"] == name), None)
            return info["value"] if info is not None else default

        inp = attr("inp", c)
        oup = attr("oup", c)
        reduction = attr("reduction", 32)
        mip = max(8, c // reduction)

        torch.manual_seed(1000 + case_index)
        conv1 = nn.Conv2d(c, mip, 1, 1, 0)
        bn1 = nn.BatchNorm2d(mip)
        conv_h = nn.Conv2d(mip, c, 1, 1, 0)
        conv_w = nn.Conv2d(mip, c, 1, 1, 0)

        params = [
            conv1.weight.detach().to(dtype), conv1.bias.detach().to(dtype),
            bn1.weight.detach().to(dtype), bn1.bias.detach().to(dtype),
            conv_h.weight.detach().to(dtype), conv_h.bias.detach().to(dtype),
            conv_w.weight.detach().to(dtype), conv_w.bias.detach().to(dtype),
        ]

        input_groups.append([x, inp, oup, reduction] + params)
    return input_groups


def get_init_inputs():
    return []
