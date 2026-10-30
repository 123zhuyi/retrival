import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class Model(nn.Module):
    """
    Model for Double Attention Network — bilinear attention with global descriptors for second-order feature interaction.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x, a=None, v=None):
        torch.manual_seed(42)
        b, c, h, w = x.shape
        if b == 0 or c == 0 or h == 0 or w == 0:
            return x
        in_channels = c
        c_m = c
        c_n = max(1, c // 8)
        reconstruct = True
        key = (in_channels, c_m, c_n, reconstruct, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            modules = [
                nn.Conv2d(c, c_m, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c_n, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c_n, 1).to(device=x.device, dtype=x.dtype)
            ]
            if reconstruct:
                modules.append(nn.Conv2d(c_m, c, 1).to(device=x.device, dtype=x.dtype))
            self._cache[key] = modules
            torch.set_rng_state(rng_state)
        convA, convB, convV, *rest = self._cache[key]

        A = convA(x).view(b, c_m, -1)
        B = F.softmax(convB(x).view(b, c_n, -1), dim=-1)
        V = F.softmax(convV(x).view(b, c_n, -1), dim=-1)
        global_descriptors = torch.bmm(A, B.permute(0,2,1))  # B,c_m,c_n
        tmpZ = torch.bmm(global_descriptors, V)  # B,c_m,h*w
        tmpZ = tmpZ.view(b, c_m, h, w)
        if reconstruct:
            tmpZ = rest[0](tmpZ)
        return tmpZ
def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "83_DoubleAttention.json")
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
        a_info = next((inp for inp in inputs if inp["name"] == "a"), None)
        a = random_tensor(a_info["shape"], dtype) if a_info else None
        v_info = next((inp for inp in inputs if inp["name"] == "v"), None)
        v = random_tensor(v_info["shape"], dtype) if v_info else None
        input_groups.append([x])
    return input_groups

def get_init_inputs(): return []