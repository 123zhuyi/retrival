import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# 参考内部计算精度升级：默认 fp32，可用环境变量 REF_COMPUTE_DTYPE=float64 覆盖。
# 权重仍按原 hash(key) 抽签在 x.dtype 下生成（hash 不变），随后无损升到 compute dtype 计算。
COMPUTE_DTYPE = getattr(torch, os.environ.get("REF_COMPUTE_DTYPE", "float32"))

class Model(nn.Module):
    """
    Model for Outlook Attention — learns attention weights from each pixel to its local neighborhood via unfold, fold, and linear projections.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x, dim, num_heads=1, kernel_size=3, padding=1, stride=1):
        torch.manual_seed(42)
        B, H, W, C = x.shape
        if B == 0 or H == 0 or W == 0 or C == 0:
            return x
        # Use compatible parameters based on actual input size
        kernel_size = max(1, kernel_size)
        stride = max(1, stride)
        padding = max(0, padding)
        if kernel_size > H or kernel_size > W:
            kernel_size = min(kernel_size, H, W)
            if kernel_size < 1:
                kernel_size = 1
        if stride > H or stride > W:
            stride = 1
        if padding >= kernel_size:
            padding = kernel_size // 2
        # Make attention dimension divisible by num_heads while preserving input channels
        attn_dim = (dim // num_heads) * num_heads
        if attn_dim == 0:
            attn_dim = num_heads
        key = (attn_dim, num_heads, kernel_size, padding, stride, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            head_dim = attn_dim // num_heads
            self._cache[key] = (
                nn.Linear(C, attn_dim).to(device=x.device, dtype=x.dtype),
                nn.Linear(C, kernel_size**4 * num_heads).to(device=x.device, dtype=x.dtype),
                nn.Linear(attn_dim, C).to(device=x.device, dtype=x.dtype)
            )
            torch.set_rng_state(rng_state)
        v_pj, attn_pj, proj = self._cache[key]

        # ===== fp32 精度升级：权重原样（升到 fp32 无损），计算链整体提到 compute dtype =====
        orig_dtype = x.dtype
        x = x.to(COMPUTE_DTYPE)
        v_pj, attn_pj, proj = (l.to(COMPUTE_DTYPE) for l in (v_pj, attn_pj, proj))

        # Compute actual unfolded grid size
        h = (H + 2 * padding - kernel_size) // stride + 1
        w = (W + 2 * padding - kernel_size) // stride + 1
        if h <= 0 or w <= 0:
            return x.to(orig_dtype)

        v = v_pj(x).permute(0,3,1,2)  # B,attn_dim,H,W
        unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        v_unfold = unfold(v)  # B, attn_dim*ks*ks, L
        L = v_unfold.shape[-1]
        head_dim = attn_dim // num_heads
        # Exact spatial layout produced by unfold
        h_out = (H + 2 * padding - kernel_size) // stride + 1
        w_out = (W + 2 * padding - kernel_size) // stride + 1
        if h_out * w_out != L:
            # Fallback: derive from L by keeping aspect ratio close to H/W
            ratio = H / max(1, W)
            w_out = int((L / ratio) ** 0.5)
            w_out = max(1, min(L, w_out))
            h_out = L // w_out
            while h_out * w_out != L and w_out > 1:
                w_out -= 1
                h_out = L // w_out
        v = v_unfold.reshape(B, num_heads, head_dim, kernel_size*kernel_size, L).permute(0,1,4,3,2)

        attn = x.permute(0,3,1,2)
        attn = F.avg_pool2d(attn, kernel_size=stride, stride=stride, ceil_mode=False)
        if attn.shape[2] != h_out or attn.shape[3] != w_out:
            attn = F.adaptive_avg_pool2d(attn, (h_out, w_out))
        attn = attn.permute(0,2,3,1)  # B,h_out,w_out,C
        attn = attn_pj(attn).reshape(B, L, num_heads, kernel_size*kernel_size, kernel_size*kernel_size).permute(0,2,1,3,4)
        scale = (head_dim ** -0.5)
        attn = F.softmax(attn * scale, dim=-1)

        out = (attn @ v).permute(0,1,4,3,2).reshape(B, attn_dim*kernel_size*kernel_size, L)
        # F.fold is the inverse of Unfold: fold back to the original spatial size
        out = F.fold(out, output_size=(H, W), kernel_size=kernel_size, padding=padding, stride=stride)
        out = proj(out.permute(0,2,3,1))  # B,H,W,C
        return out.to(orig_dtype)

def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "90_OutlookAttention.json")
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
        dim = x_info["shape"][-1]
        kernel_size_info = next((inp for inp in inputs if inp["name"] == "kernel_size"), None)
        kernel_size = kernel_size_info["value"] if kernel_size_info else None
        padding_info = next((inp for inp in inputs if inp["name"] == "padding"), None)
        padding = padding_info["value"] if padding_info else None
        stride_info = next((inp for inp in inputs if inp["name"] == "stride"), None)
        stride = stride_info["value"] if stride_info else None
        input_groups.append([x, dim, kernel_size, padding, stride])
    return input_groups

def get_init_inputs(): return []
