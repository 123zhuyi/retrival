import torch
import torch.nn as nn
import torch.fft
import json
import os

class Model(nn.Module):
    """
    Model for Global Filter Network — frequency-domain filtering via FFT, complex multiplication with learned filter, then iFFT.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x, dim=None, h=None, w=None):
        torch.manual_seed(42)
        B, N, C = x.shape
        if dim is None:
            dim = C
        if h is None and w is None:
            h = int(N ** 0.5)
            w = N // h
            while h * w != N and h > 1:
                h -= 1
                w = N // h
        elif h is None:
            h = N // w
        elif w is None:
            w = N // h
        if h * w != N:
            raise ValueError(f"GlobalFilter: h*w ({h}*{w}={h*w}) must equal N ({N})")
        dtype = x.dtype
        x = x.view(B, h, w, C).float()
        x_fft = torch.fft.rfft2(x, dim=(1,2), norm='ortho')
        # rfft2 output has shape (B, h, w//2+1, C); build filter accordingly
        r_w = x_fft.shape[2]
        key = (dim, h, r_w, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            self._cache[key] = nn.Parameter(torch.randn(h, r_w, C, 2, device=x.device, dtype=torch.float32) * 0.02)
            torch.set_rng_state(rng_state)
        weight = self._cache[key]

        weight_c = torch.view_as_complex(weight)
        x_fft = x_fft * weight_c
        x_ifft = torch.fft.irfft2(x_fft, s=(h,w), dim=(1,2), norm='ortho')
        return x_ifft.reshape(B, N, C).to(dtype)

def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "85_GlobalFilter.json")
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
        dim_info = next((inp for inp in inputs if inp["name"] == "dim"), None)
        dim = dim_info["value"] if dim_info else None
        h_info = next((inp for inp in inputs if inp["name"] == "h"), None)
        h = h_info["value"] if h_info else None
        w_info = next((inp for inp in inputs if inp["name"] == "w"), None)
        w = w_info["value"] if w_info else None
        input_groups.append([x, dim, h, w])
    return input_groups

def get_init_inputs(): return []