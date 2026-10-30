import torch.nn as nn
import torch.nn.functional as F
import json
import os
import torch, torch_npu
torch.npu.conv.allow_hf32 = False
class Model(nn.Module):
    """
    Model for Criss-Cross Attention — captures full-image contextual information through criss-cross spatial attention paths on feature maps.

    Uses self._cache for reusing dynamically created layers.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, x):
        in_dim = x.shape[1]
        torch.manual_seed(42)
        b, c, h, w = x.shape
        key = (in_dim, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            self._cache[key] = (
                nn.Conv2d(c, c//8, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c//8, 1).to(device=x.device, dtype=x.dtype),
                nn.Conv2d(c, c, 1).to(device=x.device, dtype=x.dtype),
                nn.Parameter(torch.zeros(1, device=x.device, dtype=x.dtype))
            )
            torch.set_rng_state(rng_state)
        conv_q, conv_k, conv_v, gamma = self._cache[key]

        proj_q = conv_q(x)
        proj_k = conv_k(x)
        proj_v = conv_v(x)

        # Original Criss-Cross logic
        m_batchsize = b
        proj_query_H = proj_q.permute(0,3,1,2).contiguous().view(m_batchsize*w, -1, h).permute(0,2,1)
        proj_query_W = proj_q.permute(0,2,1,3).contiguous().view(m_batchsize*h, -1, w).permute(0,2,1)
        proj_key_H = proj_k.permute(0,3,1,2).contiguous().view(m_batchsize*w, -1, h)
        proj_key_W = proj_k.permute(0,2,1,3).contiguous().view(m_batchsize*h, -1, w)
        proj_value_H = proj_v.permute(0,3,1,2).contiguous().view(m_batchsize*w, -1, h)
        proj_value_W = proj_v.permute(0,2,1,3).contiguous().view(m_batchsize*h, -1, w)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self._INF(h, x.device)).view(m_batchsize, w, h, h).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, h, w, w)

        concate = F.softmax(torch.cat([energy_H, energy_W], 3), dim=3)
        att_H = concate[:,:,:,0:h].permute(0,2,1,3).contiguous().view(m_batchsize*w, h, h)
        att_W = concate[:,:,:,h:h+w].contiguous().view(m_batchsize*h, w, w)
        out_H = torch.bmm(proj_value_H, att_H.permute(0,2,1)).view(m_batchsize, w, -1, h).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0,2,1)).view(m_batchsize, h, -1, w).permute(0,2,1,3)
        out = gamma * (out_H + out_W) + x
        return out

    def _INF(self, size, device):
        return -torch.diag(torch.tensor(float("inf"), device=device).repeat(size), 0).unsqueeze(0)
def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "81_CrissCrossAttention.json")
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