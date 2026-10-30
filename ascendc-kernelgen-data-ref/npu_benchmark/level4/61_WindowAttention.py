import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Window-based Multi-head Self-Attention (Swin/MOA-style) core，
    含相对位置偏置，投影层在算子定义内、权重作为输入传入。

    精度策略（与 Triton kernel 对齐，实测依据见 issue）：
      fp16/bf16 输入 -> 入口整体升 fp32 -> 全程 fp32 计算（qkv/scale/
      q@k^T/rel_bias/softmax/attn@v/proj）-> 输出 cast 回输入 dtype。
      原因是 kernel 侧 tl.dot 默认 fp32 累加，golden 若在 fp16/bf16
      内计算会因累加精度路径不同而被 softmax 放大成比对失败。
      fp32 输入时该路径天然等价于原始实现。

    语义与原始实现逐项对齐：
      scale = qk_scale or head_dim**-0.5（input-gen 解析后传入）；
      rel_index 为确定性常量，forward 内现场计算；
      rel_table 按原始 trunc_normal_(std=0.02) 分布生成（非零，无退化）；
      attn_drop/proj_drop 原始默认 0.0 为恒等，不纳入。
    输入约束（由 case 保证）：N == Wh * Ww，C % num_heads == 0。
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _rel_index(Wh, Ww, device):
        ch = torch.arange(Wh, device=device)
        cw = torch.arange(Ww, device=device)
        coords = torch.stack(torch.meshgrid([ch, cw], indexing='ij'))
        cf = torch.flatten(coords, 1)
        rc = cf[:, :, None] - cf[:, None, :]
        rc = rc.permute(1, 2, 0).contiguous()
        rc[:, :, 0] += Wh - 1
        rc[:, :, 1] += Ww - 1
        rc[:, :, 0] *= 2 * Ww - 1
        return rc.sum(-1)  # (Wh*Ww, Wh*Ww)

    def forward(self, x, window_size, num_heads, scale,
                qkv_w, qkv_b, proj_w, proj_b, rel_table):
        B, N, C = x.shape
        Wh, Ww = (window_size, window_size) if isinstance(window_size, int) else tuple(window_size)
        head_dim = C // num_heads

        # kernel 对齐：内部 fp32 计算
        xf = x.float()
        qkv_out = F.linear(xf, qkv_w.float(), qkv_b.float())
        qkv_out = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]

        q = q * scale
        attn = q @ k.transpose(-2, -1)

        idx = self._rel_index(Wh, Ww, x.device)
        rel_bias = rel_table.float()[idx.view(-1)].view(Wh * Ww, Wh * Ww, -1)
        rel_bias = rel_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = F.linear(out, proj_w.float(), proj_b.float())
        return out.to(x.dtype)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(shape, dtype, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    if torch.rand(1, generator=g).item() < 0.5:
        mu = float(torch.empty(1).uniform_(-5.0, 5.0, generator=g).item())
        sigma = float(torch.empty(1).uniform_(0.1, 2.0, generator=g).item())
        return torch.normal(mu, sigma, shape, dtype=torch.float32, generator=g).to(dtype)
    else:
        return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0, generator=g)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "61_WindowAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = _DTYPE_MAP[x_info["dtype"]]
        x = _random_tensor(x_info["shape"], dtype, 42 + case_index)

        dim = next((inp["value"] for inp in inputs if inp["name"] == "dim"), None)
        C = dim if dim is not None else x_info["shape"][-1]
        window_size = next((inp["value"] for inp in inputs if inp["name"] == "window_size"), 7)
        num_heads = next((inp["value"] for inp in inputs if inp["name"] == "num_heads"), 8)
        qkv_bias = next((inp["value"] for inp in inputs if inp["name"] == "qkv_bias"), True)
        qk_scale = next((inp["value"] for inp in inputs if inp["name"] == "qk_scale"), None)

        head_dim = C // num_heads
        scale = qk_scale or head_dim ** -0.5   # 与原始解析规则一致

        Wh, Ww = (window_size, window_size) if isinstance(window_size, int) else tuple(window_size)
        torch.manual_seed(1000 + case_index)
        qkv = nn.Linear(C, 3 * C, bias=True)
        proj = nn.Linear(C, C)
        rel_table = torch.empty((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        nn.init.trunc_normal_(rel_table, std=0.02)   # 与原始初始化分布一致（非零）

        qkv_b = qkv.bias.detach() if qkv_bias else torch.zeros(C * 3)  # 无 bias 语义 == 零 bias

        input_groups.append([
            x, window_size, num_heads, scale,
            qkv.weight.detach().to(dtype), qkv_b.to(dtype),
            proj.weight.detach().to(dtype), proj.bias.detach().to(dtype),
            rel_table.to(dtype),
        ])
    return input_groups


def get_init_inputs():
    return []