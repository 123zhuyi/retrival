import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

class Model(nn.Module):
    """
    Model for Halo Attention — local self-attention with halo (neighbor) blocks via block-based sliding window, using unfold for KV extraction.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x, block_size, halo_size, dim_head=64, heads=8):
        dim = x.shape[1]
        torch.manual_seed(42)
        b, c, h, w = x.shape
        if b == 0 or c == 0 or h == 0 or w == 0:
            return x
        key = (dim, block_size, halo_size, dim_head, heads, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            inner_dim = dim_head * heads
            self._cache[key] = (
                nn.Linear(dim, inner_dim, bias=False).to(device=x.device, dtype=x.dtype),
                nn.Linear(dim, inner_dim * 2, bias=False).to(device=x.device, dtype=x.dtype),
                nn.Linear(inner_dim, dim).to(device=x.device, dtype=x.dtype)
            )
            torch.set_rng_state(rng_state)
        to_q, to_kv, to_out = self._cache[key]

        block = block_size
        halo = halo_size
        # 假设输入合法
        q_inp = x.reshape(b, c, h//block, block, w//block, block).permute(0,2,4,3,5,1).reshape(b*(h//block)*(w//block), block*block, c)

        kv_inp = F.unfold(x, kernel_size=block+halo*2, stride=block, padding=halo)
        kv_inp = kv_inp.reshape(b, c, -1, kv_inp.shape[-1]).permute(0,3,2,1).reshape(b*kv_inp.shape[-1], -1, c)

        q = to_q(q_inp)
        k, v = to_kv(kv_inp).chunk(2, dim=-1)

        q = q.reshape(q.shape[0], q.shape[1], heads, -1).permute(0,2,1,3).reshape(q.shape[0]*heads, q.shape[1], -1)
        k = k.reshape(k.shape[0], k.shape[1], heads, -1).permute(0,2,1,3).reshape(k.shape[0]*heads, k.shape[1], -1)
        v = v.reshape(v.shape[0], v.shape[1], heads, -1).permute(0,2,1,3).reshape(v.shape[0]*heads, v.shape[1], -1)

        scale = dim_head ** -0.5
        q = q * scale
        sim = torch.bmm(q, k.transpose(1,2))
        # Mask out padding (not implemented here)
        attn = F.softmax(sim, dim=-1)
        out = torch.bmm(attn, v)
        out = out.reshape(-1, heads, out.shape[1], out.shape[2]).permute(0,2,1,3).reshape(-1, out.shape[1], heads*out.shape[2])
        out = to_out(out)
        out = out.reshape(b, h//block, w//block, block, block, c).permute(0,5,1,3,2,4).reshape(b,c,h,w)
        return out
def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "86_HaloAttention.json")
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
        block_size_info = next((inp for inp in inputs if inp["name"] == "block_size"), None)
        block_size = block_size_info["value"] if block_size_info else None
        halo_size_info = next((inp for inp in inputs if inp["name"] == "halo_size"), None)
        halo_size = halo_size_info["value"] if halo_size_info else None
        input_groups.append([x, block_size, halo_size])
    return input_groups

def get_init_inputs(): return []