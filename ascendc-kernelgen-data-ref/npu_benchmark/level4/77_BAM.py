import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Bottleneck Attention Module (BAM).

    Combines channel and spatial attention branches with a sigmoid gate
    and residual connection.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, x: torch.Tensor, channel: int, reduction: int = 16,
                kernel_size: int = 3, dilation_val: int = 4) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            channel: Number of input channels.
            reduction: Reduction factor for bottleneck.
            kernel_size: Kernel size for dilated convolutions in spatial branch.
            dilation_val: Dilation value for spatial branch convolutions.

        Returns:
            Attention-refined tensor of same shape as input.
        """

        b, c, h, w = x.shape
        # Ensure dilated kernel fits both spatial dimensions
        eff = dilation_val * (kernel_size - 1) + 1
        min_size = min(h, w)
        if eff > min_size:
            if min_size <= 1:
                safe_dilation = 1
                safe_kernel = 1
            else:
                safe_dilation = min(dilation_val, min_size)
                safe_kernel = min(kernel_size, (min_size - 1) // safe_dilation + 1)
                if safe_kernel > 1 and safe_kernel % 2 == 0:
                    safe_kernel -= 1
            kernel_size = safe_kernel
            dilation_val = safe_dilation

        key = (channel, reduction, kernel_size, dilation_val, x.device, x.dtype)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)

            # ---------- Channel Gate ----------
            # MLP: Linear(channel -> channel//reduction -> channel)
            mlp = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel)
            ).to(device=x.device, dtype=x.dtype)

            # ---------- Spatial Gate ----------
            # Spatial Gate: no BatchNorm to keep deterministic across runs
            conv1 = nn.Conv2d(channel, channel // reduction, kernel_size=1).to(device=x.device, dtype=x.dtype)
            conv2 = nn.Sequential(
                nn.Conv2d(channel // reduction, channel // reduction,
                          kernel_size, padding=dilation_val, dilation=dilation_val),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel // reduction,
                          kernel_size, padding=dilation_val, dilation=dilation_val),
                nn.ReLU(inplace=True)
            ).to(device=x.device, dtype=x.dtype)
            conv3 = nn.Conv2d(channel // reduction, 1, kernel_size=1).to(device=x.device, dtype=x.dtype)

            self._cache[key] = (mlp, conv1, conv2, conv3)
            torch.set_rng_state(rng_state)

        mlp, conv1, conv2, conv3 = self._cache[key]

        # ---------- Channel attention ----------
        y_ch = F.adaptive_avg_pool2d(x, 1).view(b, c)          # (b, c)
        y_ch = mlp(y_ch)                                       # (b, c)
        y_ch = y_ch.view(b, c, 1, 1)                           # (b, c, 1, 1)
        channel_attn = y_ch.expand_as(x)                       # (b, c, h, w)

        # ---------- Spatial attention ----------
        y_sp = conv1(x)                                        # (b, c//r, h, w)
        y_sp = conv2(y_sp)                                     # (b, c//r, h', w')
        y_sp = conv3(y_sp)                                     # (b, 1, h', w')
        spatial_attn = F.adaptive_avg_pool2d(y_sp, (h, w))     # (b, 1, h, w)
        spatial_attn = spatial_attn.expand_as(x)               # (b, c, h, w)

        # ---------- Combine and residual ----------
        attn = torch.sigmoid(channel_attn + spatial_attn)      # (b, c, h, w)
        out = x + x * attn
        return out


def get_input_groups():
    torch.manual_seed(42)

    json_path = os.path.join(os.path.dirname(__file__), "77_BAM.json")
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
        reduction = next((inp["value"] for inp in inputs if inp["name"] == "reduction"), 16)
        kernel_size = next((inp["value"] for inp in inputs if inp["name"] == "kernel_size"), 3)
        dilation_val = next((inp["value"] for inp in inputs if inp["name"] == "dilation_val"), 4)

        input_groups.append([x, channel, reduction, kernel_size, dilation_val])
    return input_groups


def get_init_inputs():
    return []