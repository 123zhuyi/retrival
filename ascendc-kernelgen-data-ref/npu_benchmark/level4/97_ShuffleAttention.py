import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Shuffle Attention - pure function, no trainable params.
    Input: (batch, channel, height, width)
    Combines channel attention and spatial attention within groups,
    then applies channel shuffle.
    Note: 'reduction' is kept for interface compatibility but not directly used.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    @staticmethod
    def _channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        return x.reshape(b, -1, h, w)

    def forward(self, x: torch.Tensor, channel: int, reduction: int = 16, G: int = 8) -> torch.Tensor:
        """
        Args:
            x: [batch, channel, height, width]
            channel: channel dimension (should equal x.shape[1])
            reduction: retained for interface compatibility (not used)
            G: number of groups
        Returns:
            out: [batch, channel, height, width]
        """
        torch.manual_seed(42)
        b, c, h, w = x.shape

        key = (channel, G)
        if key not in self._cache:
            torch.manual_seed(hash(key) % (2 ** 31))
            gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
            cweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
            cbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
            sweight = nn.Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
            sbias = nn.Parameter(torch.ones(1, channel // (2 * G), 1, 1))
            self._cache[key] = (gn, cweight, cbias, sweight, sbias)

        gn, cweight, cbias, sweight, sbias = self._cache[key]

        # Move params to same device and dtype as x
        device = x.device
        cweight = cweight.to(device=device, dtype=x.dtype)
        cbias = cbias.to(device=device, dtype=x.dtype)
        sweight = sweight.to(device=device, dtype=x.dtype)
        sbias = sbias.to(device=device, dtype=x.dtype)
        gn = gn.to(device=device, dtype=x.dtype)

        # Grouping: reshape into G groups
        x = x.view(b * G, -1, h, w)  # (b*G, c//G, h, w)

        # Split each group's channels into two sub-features
        x_0, x_1 = x.chunk(2, dim=1)  # each: (b*G, c//(2G), h, w)

        # ----- Channel attention branch -----
        avg_pool = F.adaptive_avg_pool2d(x_0, 1)  # (b*G, c//(2G), 1, 1)
        x_channel = cweight * avg_pool + cbias
        x_channel = x_0 * torch.sigmoid(x_channel)

        # ----- Spatial attention branch -----
        x_spatial = gn(x_1)
        x_spatial = sweight * x_spatial + sbias
        x_spatial = x_1 * torch.sigmoid(x_spatial)

        # Concatenate branches
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)

        # Channel shuffle
        out = self._channel_shuffle(out, 2)

        return out


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "97_ShuffleAttention.json")
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
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = dtype_map[x_info["dtype"]]
        x = random_tensor(x_info["shape"], dtype)

        # 解析 channel
        channel_info = next(inp for inp in inputs if inp["name"] == "channel")
        channel = channel_info["value"]

        # 解析 reduction（可能缺失，默认16）
        reduction = 16
        reduction_info = next((inp for inp in inputs if inp["name"] == "reduction"), None)
        if reduction_info is not None:
            reduction = reduction_info["value"]
        G = 8
        G_info = next((inp for inp in inputs if inp["name"] == "G"), None)
        if G_info is not None:
            G = G_info["value"]

        input_groups.append([x, channel, reduction, G])
    return input_groups


def get_init_inputs():
    return []