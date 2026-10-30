import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class Model(nn.Module):
    """
    Model for Sequential Polarized Self-Attention — channel attention followed by spatial attention sequentially, with LayerNorm.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x, mix_weight=None, use_channel=None, scale_factor=None):
        channel = x.shape[1]
        torch.manual_seed(42)
        b, c, h, w = x.shape
        if b == 0 or c == 0 or h == 0 or w == 0:
            return x
        c2 = max(1, c // 2)
        key = (channel, c2, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            self._cache[key] = (
                nn.Conv2d(c, c2, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, 1, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c2, c, 1).to(device=x.device, dtype=x.dtype),
                nn.LayerNorm(c).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c2, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c2, 1).to(device=x.device, dtype=x.dtype),
                nn.AdaptiveAvgPool2d((1,1))
            )
            torch.set_rng_state(rng_state)
        ch_wv, ch_wq, ch_wz, ln, sp_wv, sp_wq, agp = self._cache[key]

        # Channel attention
        ch_wv_out = ch_wv(x).view(b, c2, -1)
        ch_wq_out = ch_wq(x).view(b, -1, 1)
        ch_wq_out = F.softmax(ch_wq_out, dim=1)
        ch_wz_out = torch.matmul(ch_wv_out, ch_wq_out).unsqueeze(-1)
        ch_wz_out = ch_wz(ch_wz_out).view(b, c, 1).permute(0,2,1)
        ch_weight = torch.sigmoid(ln(ch_wz_out)).permute(0,2,1).view(b, c, 1, 1)
        channel_out = ch_weight * x

        # Spatial attention on channel_out
        sp_wv_out = sp_wv(channel_out).view(b, c2, -1)
        sp_wq_out = agp(sp_wq(channel_out)).view(b, -1, 1)
        sp_wq_out = F.softmax(sp_wq_out.permute(0,2,1), dim=-1)
        sp_wz_out = torch.matmul(sp_wq_out, sp_wv_out).view(b, 1, h, w)
        sp_weight = torch.sigmoid(sp_wz_out)
        spatial_out = sp_weight * channel_out

        return spatial_out
def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "96_SequentialPolarizedSelfAttention.json")
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
        mix_weight_info = next((inp for inp in inputs if inp["name"] == "mix_weight"), None)
        mix_weight = mix_weight_info["value"] if mix_weight_info else None
        use_channel_info = next((inp for inp in inputs if inp["name"] == "use_channel"), None)
        use_channel = use_channel_info["value"] if use_channel_info else None
        scale_factor_info = next((inp for inp in inputs if inp["name"] == "scale_factor"), None)
        scale_factor = scale_factor_info["value"] if scale_factor_info else None
        input_groups.append([x, mix_weight, use_channel, scale_factor])
    return input_groups

def get_init_inputs(): return []