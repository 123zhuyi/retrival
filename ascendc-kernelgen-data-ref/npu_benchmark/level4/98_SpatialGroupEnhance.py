import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    SpatialGroupEnhance (SGE): Spatial Group-wise Enhance module.

    Groups channels and enhances spatial attention within each group by
    computing similarity between each spatial position and the global
    average-pooled feature, followed by normalization and sigmoid gating.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            groups: Number of channel groups. Must divide channels.

        Returns:
            Enhanced tensor of same shape as input.
        """
        torch.manual_seed(42)
        b, c, h, w = x.shape
        key = (groups, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            weight = nn.Parameter(torch.zeros(1, groups, 1, 1, device=x.device, dtype=x.dtype))
            bias = nn.Parameter(torch.zeros(1, groups, 1, 1, device=x.device, dtype=x.dtype))
            self._cache[key] = (weight, bias)
            torch.set_rng_state(rng_state)
        weight, bias = self._cache[key]

        x_reshaped = x.view(b * groups, -1, h, w)          # (b*g, c//g, h, w)
        avg_pool = F.adaptive_avg_pool2d(x_reshaped, 1)    # (b*g, c//g, 1, 1)
        xn = x_reshaped * avg_pool                         # (b*g, c//g, h, w)
        xn = xn.sum(dim=1, keepdim=True)                   # (b*g, 1, h, w)
        t = xn.view(b * groups, -1)                        # (b*g, h*w)
        t = t - t.mean(dim=1, keepdim=True)                # (b*g, h*w)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std                                        # (b*g, h*w)
        t = t.view(b, groups, h, w)                        # (b, g, h, w)
        t = t * weight + bias                              # (b, g, h, w)
        t = t.view(b * groups, 1, h, w)                    # (b*g, 1, h, w)
        out = x_reshaped * torch.sigmoid(t)                # (b*g, c//g, h, w)
        out = out.view(b, c, h, w)
        return out


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "98_SpatialGroupEnhance.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
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

        groups = next(inp for inp in inputs if inp["name"] == "groups")["value"]
        input_groups.append([x, groups])
    return input_groups


def get_init_inputs():
    return []