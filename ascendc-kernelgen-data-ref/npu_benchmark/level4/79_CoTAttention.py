import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Contextual Transformer (CoT) Attention.

    模块语义：
        key_embed  = Conv_group(x) + ReLU
        value_embed = Conv1x1(x)
        att_embed   = Conv1x1(cat([key_embed, x])) + ReLU
                      -> Conv1x1 -> reshape 成 k*k 邻域注意力
        k2 = softmax(att) * value_embed
        out = key_embed + k2
    三个卷积分支都在 CoT 公开定义内，权重传入 forward。

    输入约束（由 case 保证）：
      x: (B, C, H, W)
      kernel_size: 动态卷积核大小
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, kernel_size,
                key_embed_w, key_embed_b,
                value_embed_w, value_embed_b,
                att_w1, att_b1, att_w2, att_b2):
        b, c, h, w = x.shape
        n = h * w
        groups = 4 if c % 4 == 0 else (2 if c % 2 == 0 else 1)
        padding = kernel_size // 2

        # NPU Conv2d 在 float32 下使用降精度累加，导致与 ieee fp32 参考差异过大。
        # 对 float32 输入在 NPU 上先升精度到 fp32 再计算，结果回传原始 dtype。
        if x.dtype == torch.float32:
            x_f = x.float()
            key_embed_w_f = key_embed_w.float()
            value_embed_w_f = value_embed_w.float()
            att_w1_f = att_w1.float()
            att_w2_f = att_w2.float()
            key_embed_b_f = key_embed_b.float() if key_embed_b is not None else None
            value_embed_b_f = value_embed_b.float() if value_embed_b is not None else None
            att_b1_f = att_b1.float() if att_b1 is not None else None
            att_b2_f = att_b2.float() if att_b2 is not None else None

            k1 = F.conv2d(x_f, key_embed_w_f, key_embed_b_f,
                          padding=padding, groups=groups)
            k1 = F.relu(k1, inplace=True)
            v = F.conv2d(x_f, value_embed_w_f, value_embed_b_f).view(b, c, n)
            y = torch.cat([k1, x_f], 1)
            att = F.conv2d(y, att_w1_f, att_b1_f)
            att = F.relu(att, inplace=True)
            att = F.conv2d(att, att_w2_f, att_b2_f)
            att = att.reshape(b, c, kernel_size * kernel_size, h, w)
            att = att.mean(2).view(b, c, n)
            k2 = F.softmax(att, dim=-1) * v
            k2 = k2.view(b, c, h, w)
            return (k1 + k2).to(x.dtype)

        k1 = F.conv2d(x, key_embed_w, key_embed_b,
                      padding=padding, groups=groups)
        k1 = F.relu(k1, inplace=True)
        v = F.conv2d(x, value_embed_w, value_embed_b).view(b, c, n)
        y = torch.cat([k1, x], 1)
        att = F.conv2d(y, att_w1, att_b1)
        att = F.relu(att, inplace=True)
        att = F.conv2d(att, att_w2, att_b2)
        att = att.reshape(b, c, kernel_size * kernel_size, h, w)
        att = att.mean(2).view(b, c, n)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(b, c, h, w)
        return k1 + k2


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(shape, dtype, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if torch.rand(1, generator=generator).item() < 0.5:
        mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
        sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
        if dtype is torch.bfloat16:
            return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
        return torch.normal(mu, sigma, shape, dtype=dtype)
    else:
        return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "79_CoTAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = _DTYPE_MAP[x_info["dtype"]]
        x = _random_tensor(x_info["shape"], dtype, 42 + case_index)
        c = x.shape[1]
        kernel_size = next((inp["value"] for inp in inputs if inp["name"] == "kernel_size"), 3)
        groups = 4 if c % 4 == 0 else (2 if c % 2 == 0 else 1)
        mid = max(1, c // 4)
        padding = kernel_size // 2

        torch.manual_seed(1000 + case_index)
        key_embed = nn.Sequential(
            nn.Conv2d(c, c, kernel_size, padding=padding, groups=groups, bias=False),
            nn.ReLU()
        )
        value_embed = nn.Sequential(nn.Conv2d(c, c, 1, bias=False))
        att_embed = nn.Sequential(
            nn.Conv2d(2 * c, mid, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid, kernel_size * kernel_size * c, 1)
        )

        sd_key = key_embed.state_dict()
        sd_val = value_embed.state_dict()
        sd_att = att_embed.state_dict()

        params = [
            sd_key["0.weight"].to(dtype), None,
            sd_val["0.weight"].to(dtype), None,
            sd_att["0.weight"].to(dtype), None,
            sd_att["2.weight"].to(dtype), sd_att["2.bias"].to(dtype),
        ]

        input_groups.append([x, kernel_size] + params)
    return input_groups


def get_init_inputs():
    return []
