import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    模块语义（与原始算子严格一致）：
        1) 通道注意力：avg-pool 和 max-pool 两路分别经过共享 MLP
           （Conv2d C->C/reduction -> ReLU -> Conv2d C/reduction->C，均无 bias），
           相加后过 sigmoid 得到通道权重，与 x 相乘；
        2) 空间注意力：对通道加权后的特征沿通道取 mean/max，拼接成 2 通道，
           经过 Conv2d(2->1, kernel_size, padding=kernel_size//2，含 bias)，
           过 sigmoid 得到空间权重，再相乘；
        3) 输出 = 加权结果 + 残差（原始输入 x）。

    输入约束（由 case 保证）：
      x: (B, C, H, W)
      channel: 保留用于接口兼容，实际使用 x.shape[1]
      reduction: MLP 瓶颈降维比
      kernel_size: 空间注意力 2D 卷积核大小
      ca_fc1_weight: (C//reduction, C, 1, 1)，通道注意力第一层卷积权重
      ca_fc2_weight: (C, C//reduction, 1, 1)，通道注意力第二层卷积权重
      sa_conv_weight: (1, 2, kernel_size, kernel_size)，空间注意力卷积权重
      sa_conv_bias: (1,)，空间注意力卷积偏置
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, channel, reduction, kernel_size,
                ca_fc1_weight, ca_fc2_weight, sa_conv_weight, sa_conv_bias):
        residual = x

        # ---- Channel Attention ----
        avg_p = F.adaptive_avg_pool2d(x, 1)   # (B, C, 1, 1)
        max_p = F.adaptive_max_pool2d(x, 1)   # (B, C, 1, 1)

        def shared_mlp(v):
            v = F.conv2d(v, ca_fc1_weight)          # C -> C//reduction
            v = F.relu(v)
            v = F.conv2d(v, ca_fc2_weight)          # C//reduction -> C
            return v

        channel_att = torch.sigmoid(shared_mlp(avg_p) + shared_mlp(max_p))
        out = x * channel_att

        # ---- Spatial Attention ----
        avg_out = torch.mean(out, dim=1, keepdim=True)      # (B, 1, H, W)
        max_out, _ = torch.max(out, dim=1, keepdim=True)    # (B, 1, H, W)
        spatial_in = torch.cat([max_out, avg_out], dim=1)   # (B, 2, H, W)
        spatial_att = torch.sigmoid(
            F.conv2d(spatial_in, sa_conv_weight, sa_conv_bias,
                     padding=kernel_size // 2))
        out = out * spatial_att

        # ---- Residual ----
        return out + residual


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
        sigma = float(torch.empty(1).uniform_(0.5, 2.0).item())
        if dtype is torch.bfloat16:
            return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
        return torch.normal(mu, sigma, shape, dtype=dtype)
    else:
        return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "78_CBAMBlock.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = _DTYPE_MAP[x_info["dtype"]]
        x = _random_tensor(x_info["shape"], dtype, 42 + case_index)

        def attr(name, default=None):
            info = next((i for i in inputs if i["name"] == name), None)
            return info["value"] if info is not None else default

        channel = attr("channel", x.shape[1])
        reduction = attr("reduction", 16)
        kernel_size = attr("kernel_size", 49)

        hidden = max(1, channel // reduction)

        torch.manual_seed(1000 + case_index)
        # 通道注意力共享 MLP 的两层 1x1 卷积（无 bias）
        ca_fc1 = nn.Conv2d(channel, hidden, 1, bias=False)
        ca_fc2 = nn.Conv2d(hidden, channel, 1, bias=False)
        # 空间注意力 2->1 卷积（含 bias）
        sa_conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                            padding=kernel_size // 2, bias=True)

        ca_fc1_weight = ca_fc1.weight.detach().to(dtype=dtype) * 0.1
        ca_fc2_weight = ca_fc2.weight.detach().to(dtype=dtype) * 0.1
        sa_conv_weight = sa_conv.weight.detach().to(dtype=dtype) * 0.1
        sa_conv_bias = sa_conv.bias.detach().to(dtype=dtype) * 0.1

        input_groups.append([
            x, channel, reduction, kernel_size,
            ca_fc1_weight, ca_fc2_weight, sa_conv_weight, sa_conv_bias,
        ])
    return input_groups


def get_init_inputs():
    return []
