import torch.nn as nn
import torch.nn.functional as F
import json
import os
import torch, torch_npu
torch.npu.conv.allow_hf32 = False

class Model(nn.Module):
    """
    GCNet Global Context Module（KernelBench 参考实现版）。

    模块语义（严格遵循源码实际行为）：
        attn = Conv1x1(x)                      # (B,1,H,W) raw logits
        ※ 源码 __init__ 定义了 nn.Softmax(dim=2) 但 context_modeling 从未调用，
          golden 遵循源码：不加 softmax，直接用 raw logits 做加权和。
          （GCNet 论文/官方实现有 softmax，此处是有意复刻参考代码的行为）
        context = x_flat @ attn^T              # (B,C,1,1)，未归一化加权和
        out = x + Transform(context)
    Transform = Conv1x1(C->C//r) -> LN -> ReLU -> Conv1x1(C//r->C)。
    所有 Conv/LN 在模块定义内，权重传入 forward。
    LayerNorm 只用到传入的 weight/bias 做仿射（无 running stats）。

    输入约束（由 case 保证）：
      x: (B, C, H, W)
      channel: 等于 C
      reduction: bottleneck 缩减比
      context: 保留接口兼容，未使用
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, channel, reduction, context,
                conv1_w, conv1_b,
                t_w1, t_b1, ln_w, ln_b, t_w2, t_b2):
        b, c, h, w = x.shape

        attn = F.conv2d(x, conv1_w, conv1_b)
        b_att, _, h_att, w_att = attn.shape
        attn = attn.view(b_att, 1, h_att * w_att)
        # 注意：源码未调用其定义的 softmax，此处不得加归一化
        x_flat = x.view(b, c, -1)
        context_computed = torch.bmm(x_flat, attn.transpose(1, 2))
        context_computed = context_computed.unsqueeze(-1)

        y = F.conv2d(context_computed, t_w1, t_b1)
        y = F.layer_norm(y, y.shape[1:], ln_w, ln_b)
        y = F.relu(y, inplace=True)
        y = F.conv2d(y, t_w2, t_b2)

        out = x + y
        return out


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(shape, dtype, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if torch.rand(1, generator=generator).item() < 0.5:
        mu = float(torch.empty(1).uniform_(-5.0, 5.0, generator=generator).item())
        sigma = float(torch.empty(1).uniform_(0.1, 2.0, generator=generator).item())
        if dtype is torch.bfloat16:
            return torch.normal(
                mu, sigma, shape, dtype=torch.float32, generator=generator
            ).to(dtype)
        return torch.normal(mu, sigma, shape, dtype=dtype, generator=generator)
    else:
        return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0, generator=generator)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "84_GCModule.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = _DTYPE_MAP[x_info["dtype"]]
        x = _random_tensor(x_info["shape"], dtype, 42 + case_index)
        c = x.shape[1]

        channel_info = next(inp for inp in inputs if inp["name"] == "channel")
        channel = channel_info["value"]
        # 源码 __init__(channel, reduction) 内 Conv 均以 channel 为输入通道数
        assert channel == c, f"case {case_index}: channel({channel}) != C({c})"
        reduction_info = next(inp for inp in inputs if inp["name"] == "reduction")
        reduction = reduction_info["value"]

        torch.manual_seed(1000 + case_index)
        conv1 = nn.Conv2d(c, 1, kernel_size=1)
        transform = nn.Sequential(
            nn.Conv2d(c, c // reduction, kernel_size=1),
            nn.LayerNorm([c // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // reduction, c, kernel_size=1)
        )

        sd_conv1 = conv1.state_dict()
        sd_t = transform.state_dict()
        params = [
            sd_conv1["weight"].to(dtype), sd_conv1["bias"].to(dtype),
            sd_t["0.weight"].to(dtype), sd_t["0.bias"].to(dtype),
            sd_t["1.weight"].to(dtype), sd_t["1.bias"].to(dtype),
            sd_t["3.weight"].to(dtype), sd_t["3.bias"].to(dtype),
        ]

        context = None
        for inp in inputs:
            if inp.get("name") == "context" and inp["shape"] is not None:
                context = _random_tensor(inp["shape"], dtype, 42 + case_index + 10)

        input_groups.append([x, channel, reduction, context] + params)
    return input_groups


def get_init_inputs():
    return []