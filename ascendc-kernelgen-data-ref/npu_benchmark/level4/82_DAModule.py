import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class Model(nn.Module):
    """
    Model for Dual Attention Module — combines position attention and channel attention branches in parallel with softmax.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x, kernel_size=3):
        channel = x.shape[1]
        torch.manual_seed(42)
        b, c, h, w = x.shape
        key = (channel, kernel_size, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            conv_pos = nn.Conv2d(c, c, kernel_size, padding=kernel_size//2).to(device=x.device, dtype=x.dtype)
            conv_ch = nn.Conv2d(c, c, kernel_size, padding=kernel_size//2).to(device=x.device, dtype=x.dtype)
            self._cache[key] = (conv_pos, conv_ch)
            torch.set_rng_state(rng_state)
        conv_pos, conv_ch = self._cache[key]

        # Position attention
        y_pos = conv_pos(x).view(b, c, -1).permute(0, 2, 1)  # B, N, C
        y_pos_f = y_pos.float()
        attn_pos = F.softmax(torch.bmm(y_pos_f, y_pos_f.transpose(1, 2)) * (c ** -0.5), dim=-1)
        out_pos = torch.bmm(attn_pos, y_pos_f).permute(0, 2, 1).view(b, c, h, w).to(x.dtype)

        # Channel attention
        y_ch = conv_ch(x).view(b, c, -1)  # B, C, N
        y_ch_f = y_ch.float()
        attn_ch = F.softmax(torch.bmm(y_ch_f, y_ch_f.transpose(1, 2)) * ((h*w) ** -0.5), dim=-1)
        out_ch = torch.bmm(attn_ch, y_ch_f).view(b, c, h, w).to(x.dtype)

        return out_pos + out_ch
def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "82_DAModule.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = dtype_map[x_info["dtype"]]
        x = random_tensor(x_info["shape"], dtype)
        input_groups.append([x])
    return input_groups

def get_init_inputs(): return []