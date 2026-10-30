import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Axial Attention — 沿高度轴和宽度轴分别做 self-attention，结果相加。

    模块语义：
        to_q / to_kv / to_out 为模块内部 Linear；
        分别对 H 轴、W 轴计算 scaled-dot attention；
        输出相加后返回 (B, C, H, W)。
    所有 Linear 在模块定义内，权重传入 forward。

    输入约束（由 case 保证）：
      x: (B, C, H, W)
      dim, heads, dim_heads: 超参
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        dim,
        heads,
        dim_heads,
        to_q_w,
        to_q_b,
        to_kv_w,
        to_kv_b,
        to_out_w,
        to_out_b,
    ):
        b, c, h, w = x.shape
        if dim is None:
            dim = c
        if heads is None:
            heads = 8
        head_dim = (dim // heads) if dim_heads is None else dim_heads
        inner_dim = head_dim * heads

        x_perm = x.permute(0, 2, 3, 1).contiguous()  # B,H,W,C
        out = torch.zeros_like(x_perm)

        # Height axis
        seq_h = x_perm.permute(0, 2, 1, 3).contiguous().view(b * w, h, c)
        q_h = F.linear(seq_h, to_q_w, to_q_b)
        kv_h = F.linear(seq_h, to_kv_w, to_kv_b)
        k_h, v_h = kv_h.chunk(2, dim=-1)
        q_h = q_h.view(q_h.shape[0], q_h.shape[1], heads, -1).transpose(1, 2)
        k_h = k_h.view(k_h.shape[0], k_h.shape[1], heads, -1).transpose(1, 2)
        v_h = v_h.view(v_h.shape[0], v_h.shape[1], heads, -1).transpose(1, 2)
        attn_h = F.softmax(
            torch.matmul(q_h, k_h.transpose(-2, -1)) * (head_dim ** -0.5),
            dim=-1,
        )
        out_h = torch.matmul(attn_h, v_h).transpose(1, 2).contiguous().view(
            seq_h.shape[0], seq_h.shape[1], -1
        )
        out_h = F.linear(out_h, to_out_w, to_out_b).view(b, w, h, c).permute(0, 2, 1, 3)

        # Width axis
        seq_w = x_perm.view(b * h, w, c)
        q_w = F.linear(seq_w, to_q_w, to_q_b)
        kv_w = F.linear(seq_w, to_kv_w, to_kv_b)
        k_w, v_w = kv_w.chunk(2, dim=-1)
        q_w = q_w.view(q_w.shape[0], q_w.shape[1], heads, -1).transpose(1, 2)
        k_w = k_w.view(k_w.shape[0], k_w.shape[1], heads, -1).transpose(1, 2)
        v_w = v_w.view(v_w.shape[0], v_w.shape[1], heads, -1).transpose(1, 2)
        attn_w = F.softmax(
            torch.matmul(q_w, k_w.transpose(-2, -1)) * (head_dim ** -0.5),
            dim=-1,
        )
        out_w = torch.matmul(attn_w, v_w).transpose(1, 2).contiguous().view(
            seq_w.shape[0], seq_w.shape[1], -1
        )
        out_w = F.linear(out_w, to_out_w, to_out_b).view(b, h, w, c)

        out = out_h + out_w
        return out.permute(0, 3, 1, 2).contiguous()


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(shape, dtype, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if torch.rand(1, generator=generator).item() < 0.5:
        mu = float((torch.rand(1, generator=generator) * 0.6 - 0.3).item())
        sigma = float((0.1 + torch.rand(1, generator=generator) * 0.2).item())
        if dtype is torch.bfloat16:
            return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
        return torch.normal(mu, sigma, shape, dtype=dtype)
    else:
        return (torch.rand(shape, generator=generator) * 0.6 - 0.3).to(dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "76_AxialAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = dtype_map[x_info["dtype"]]
        x = _random_tensor(x_info["shape"], dtype, 42 + case_index)

        dim_info = next((inp for inp in inputs if inp["name"] == "dim"), None)
        dim = dim_info["value"] if dim_info else None
        heads_info = next((inp for inp in inputs if inp["name"] == "heads"), None)
        heads = heads_info["value"] if heads_info else None
        dim_heads_info = next(
            (inp for inp in inputs if inp["name"] == "dim_heads"), None
        )
        dim_heads = dim_heads_info["value"] if dim_heads_info else None

        c = x.shape[1]
        effective_dim = dim if dim is not None else c
        effective_heads = heads if heads is not None else 8
        head_dim = (
            (effective_dim // effective_heads)
            if dim_heads is None
            else dim_heads
        )
        inner_dim = head_dim * effective_heads

        torch.manual_seed(1000 + case_index)
        to_q = nn.Linear(effective_dim, inner_dim, bias=False)
        to_kv = nn.Linear(effective_dim, 2 * inner_dim, bias=False)
        to_out = nn.Linear(inner_dim, effective_dim)

        input_groups.append([
            x, dim, heads, dim_heads,
            to_q.weight.detach().to(dtype), None,
            to_kv.weight.detach().to(dtype), None,
            to_out.weight.detach().to(dtype), to_out.bias.detach().to(dtype),
        ])
    return input_groups


def get_init_inputs():
    return []
