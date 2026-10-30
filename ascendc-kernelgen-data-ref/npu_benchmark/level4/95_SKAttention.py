import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from collections import OrderedDict


class Model(nn.Module):
    """
    Selective Kernel Attention (SK Attention).

    Uses multiple convolutional kernels of different sizes and fuses their
    outputs via a softmax attention mechanism for adaptive receptive field selection.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, x: torch.Tensor, channel: int, kernels: list,
                reduction: int = 16, group: int = 1, L: int = 32) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            channel: Number of input channels.
            kernels: List of kernel sizes (e.g., [1, 3, 5, 7]).
            reduction: Reduction factor for the bottleneck.
            group: Group convolution parameter.
            L: Minimum dimension for the fc layer.

        Returns:
            Attention-fused tensor of same shape as input.
        """
        torch.manual_seed(42)
        b, c, h, w = x.shape
        if channel is None:
            channel = c
        # kernels must be tuple for hashing
        key = (channel, tuple(kernels), reduction, group, L, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)

            d = max(L, channel // reduction)
            # Build convs
            convs = nn.ModuleList()
            for k in kernels:
                convs.append(
                    nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(channel, channel, kernel_size=k,
                                           padding=k // 2, groups=group).to(device=x.device, dtype=x.dtype)),
                        ('bn', nn.BatchNorm2d(channel).to(device=x.device, dtype=x.dtype)),
                        ('relu', nn.ReLU())
                    ]))
                )
            # fc layer
            fc = nn.Linear(channel, d).to(device=x.device, dtype=x.dtype)
            # fcs layers
            fcs = nn.ModuleList()
            for _ in kernels:
                fcs.append(nn.Linear(d, channel).to(device=x.device, dtype=x.dtype))

            self._cache[key] = (convs, fc, fcs)
            torch.set_rng_state(rng_state)

        convs, fc, fcs = self._cache[key]

        # ---------- Split ----------
        conv_outs = []
        for conv in convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # (k, b, c, h, w)

        # ---------- Fuse ----------
        U = sum(conv_outs)  # (b, c, h, w)

        # ---------- Reduction ----------
        S = U.mean(-1).mean(-1)  # (b, c)
        Z = fc(S)                # (b, d)

        # ---------- Attention weights ----------
        weights = []
        for fc_i in fcs:
            w = fc_i(Z).view(b, c, 1, 1)  # (b, c, 1, 1)
            weights.append(w)
        attention_weights = torch.stack(weights, 0)  # (k, b, c, 1, 1)
        attention_weights = F.softmax(attention_weights, dim=0)  # softmax along kernel dimension

        # ---------- Fuse ----------
        V = (attention_weights * feats).sum(0)  # (b, c, h, w)
        return V


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "95_SKAttention.json")
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

        channel = next(inp for inp in inputs if inp["name"] == "channel")["value"]
        kernels = next(inp for inp in inputs if inp["name"] == "kernels")["value"]  # list of ints
        reduction = next((inp["value"] for inp in inputs if inp["name"] == "reduction"), 16)
        group = next((inp["value"] for inp in inputs if inp["name"] == "group"), 1)
        L = next((inp["value"] for inp in inputs if inp["name"] == "L"), 32)

        input_groups.append([x, channel, kernels, reduction, group, L])
    return input_groups


def get_init_inputs():
    return []