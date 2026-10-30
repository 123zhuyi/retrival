import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import zlib

def spatial_shift1(x):
    b, w, h, c = x.shape
    out = x.clone()
    out[:, 1:, :, :c//4] = x[:, :w-1, :, :c//4]
    out[:, :w-1, :, c//4:c//2] = x[:, 1:, :, c//4:c//2]
    out[:, :, 1:, c//2:c*3//4] = x[:, :, :h-1, c//2:c*3//4]
    out[:, :, :h-1, 3*c//4:] = x[:, :, 1:, 3*c//4:]
    return out

def spatial_shift2(x):
    b, w, h, c = x.shape
    out = x.clone()
    out[:, :, 1:, :c//4] = x[:, :, :h-1, :c//4]
    out[:, :, :h-1, c//4:c//2] = x[:, :, 1:, c//4:c//2]
    out[:, 1:, :, c//2:c*3//4] = x[:, :w-1, :, c//2:c*3//4]
    out[:, :w-1, :, 3*c//4:] = x[:, 1:, :, 3*c//4:]
    return out

class Model(nn.Module):
    """
    Model for S2 (Spatial Shift + Split) Attention — spatial shift operations followed by split attention with MLPs.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x):
        channels = x.shape[1]
        b, c, h, w = x.shape
        if b == 0 or c == 0 or h == 0 or w == 0:
            return x
        key = (channels, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed((zlib.crc32(repr(key).encode()) + 4) & 0xFFFFFFFF)
            self._cache[key] = (
                nn.Linear(c, 3*c).to(device=x.device, dtype=x.dtype),
                nn.Linear(c, c).to(device=x.device, dtype=x.dtype),
                nn.Linear(c, c).to(device=x.device, dtype=x.dtype),
                nn.Linear(c, 3*c).to(device=x.device, dtype=x.dtype)
            )
            torch.set_rng_state(rng_state)
        mlp1, mlp2, mlp3, mlp4 = self._cache[key]

        x = x.permute(0,3,2,1).contiguous()  # B,C,H,W -> B,W,H,C
        x = mlp1(x)  # B,W,H,3C
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:2*c])
        x3 = x[:,:,:,2*c:]
        x_all = torch.stack([x1,x2,x3], 1)  # B,3,W,H,C

        # SplitAttention
        B, k, W, H, C_ = x_all.shape
        x_all = x_all.view(B, k, -1, C_)  # B,3,N,C
        # Average over branches and tokens in fp32 to avoid fp16 overflow on large N
        a = x_all.float().mean(dim=(1, 2)).to(x.dtype)  # B,C
        a = mlp3(a)
        a = F.gelu(a)
        a = mlp4(a).view(B, 3, C_)  # B,3,C
        a = a.float().softmax(dim=1).to(x.dtype)  # B,3,C
        out = (a.unsqueeze(-2) * x_all).sum(1).view(B, W, H, C_)  # B,W,H,C
        out = mlp2(out)
        out = out.permute(0,3,2,1)  # B,W,H,C -> B,C,H,W
        return out
def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "94_S2Attention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # S2 attention expands channels by 3x and applies GeLU/softmax; use a smaller
        # range for fp16/bf16 to avoid intermediate overflow.
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-1.0, 1.0).item())
            sigma = float(torch.empty(1).uniform_(0.05, 0.2).item())
            if dtype is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-1.0, 1.0)

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