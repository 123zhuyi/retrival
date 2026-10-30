import torch
from torch import nn
import json
import os


class Model(nn.Module):
    """
    Weighted Permute MLP (ViP) that applies separate linear projections along
    the channel, height, and width dimensions, then combines them using
    learned reweighting coefficients.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, x: torch.Tensor, dim: int, seg_dim: int,
                qkv_bias: bool = False, proj_drop: float = 0.0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, height, width, channels).
            dim: Feature dimension.
            seg_dim: Segmentation dimension (group size for permute).
            qkv_bias: Whether to use bias in linear layers.
            proj_drop: Dropout rate for projection.

        Returns:
            Output tensor of same shape as input.
        """
        torch.manual_seed(42)
        B, H, W, C = x.shape
        if B == 0 or H == 0 or W == 0 or C == 0:
            return x
        # Infer from actual input; JSON values may not match or divide cleanly
        dim = C
        # Pick a seg_dim that divides both C and H*W
        seg_dim = min(seg_dim, C, H * W)
        seg_dim = max(1, seg_dim)
        while C % seg_dim != 0 or (H * W) % seg_dim != 0:
            seg_dim -= 1
            if seg_dim <= 1:
                seg_dim = 1
                break

        S = C // seg_dim
        # Business key for initialization (no device/dtype)
        business_key = (dim, seg_dim, H, W, qkv_bias, proj_drop)
        cache_key = (business_key, x.device, x.dtype)

        if cache_key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(business_key) & 0xFFFFFFFF)

            # Three linear projections along C, H, W
            mlp_c = nn.Linear(dim, dim, bias=qkv_bias).to(device=x.device, dtype=x.dtype)
            # H- and W- MLPs operate on the flattened spatial-group dimensions
            mlp_h = nn.Linear(H * S, H * S, bias=qkv_bias).to(device=x.device, dtype=x.dtype)
            mlp_w = nn.Linear(W * S, W * S, bias=qkv_bias).to(device=x.device, dtype=x.dtype)

            # MLP for reweighting (dim -> dim//4 -> dim*3)
            hidden = dim // 4
            # MLP layers: fc1, act, fc2, drop (drop=0.0)
            mlp_fc1 = nn.Linear(dim, hidden).to(device=x.device, dtype=x.dtype)
            mlp_act = nn.GELU()
            mlp_fc2 = nn.Linear(hidden, dim * 3).to(device=x.device, dtype=x.dtype)
            mlp_drop = nn.Dropout(0.0)  # no parameters, no need to .to but we keep consistent
            # We'll store mlp components separately to avoid nested object issues

            # Projection and dropout
            proj = nn.Linear(dim, dim).to(device=x.device, dtype=x.dtype)
            proj_drop_layer = nn.Dropout(proj_drop)  # no parameters

            self._cache[cache_key] = (mlp_c, mlp_h, mlp_w,
                                      mlp_fc1, mlp_act, mlp_fc2, mlp_drop,
                                      proj, proj_drop_layer)
            torch.set_rng_state(rng_state)

        (mlp_c, mlp_h, mlp_w,
         mlp_fc1, mlp_act, mlp_fc2, mlp_drop,
         proj, proj_drop_layer) = self._cache[cache_key]

        # ---------- c_embed ----------
        c_embed = mlp_c(x)  # (B, H, W, C)

        # ---------- h_embed ----------
        # Reshape: (B, H, W, seg_dim, S) -> permute(0,3,2,1,4) -> (B, seg_dim, W, H*S)
        h_embed = x.reshape(B, H, W, seg_dim, S).permute(0, 3, 2, 1, 4).contiguous()
        h_embed = h_embed.reshape(B, seg_dim, W, H * S)
        h_embed = mlp_h(h_embed)  # (B, seg_dim, W, H*S)
        h_embed = h_embed.reshape(B, seg_dim, W, H, S).permute(0, 3, 2, 1, 4).contiguous()
        h_embed = h_embed.reshape(B, H, W, C)

        # ---------- w_embed ----------
        w_embed = x.reshape(B, H, W, seg_dim, S).permute(0, 3, 1, 2, 4).contiguous()
        w_embed = w_embed.reshape(B, seg_dim, H, W * S)
        w_embed = mlp_w(w_embed)  # (B, seg_dim, H, W*S)
        w_embed = w_embed.reshape(B, seg_dim, H, W, S).permute(0, 2, 3, 1, 4).contiguous()
        w_embed = w_embed.reshape(B, H, W, C)

        # ---------- Reweighting ----------
        # weight = (c_embed + h_embed + w_embed).permute(0,3,1,2).flatten(2).mean(2)  # (B, C)
        weight = (c_embed + h_embed + w_embed).permute(0, 3, 1, 2).contiguous()
        weight = weight.flatten(2).mean(dim=2)  # (B, C)

        # MLP forward: fc1 -> act -> dropout -> fc2 -> dropout
        weight = mlp_fc1(weight)
        weight = mlp_act(weight)
        weight = mlp_drop(weight)
        weight = mlp_fc2(weight)
        weight = mlp_drop(weight)  # (B, C*3)
        weight = weight.reshape(B, C, 3).permute(2, 0, 1)  # (3, B, C)
        weight = weight.softmax(dim=0)  # (3, B, C)
        weight = weight.unsqueeze(2).unsqueeze(2)  # (3, B, C, 1, 1)

        # Combine
        x_out = c_embed * weight[0] + w_embed * weight[1] + h_embed * weight[2]  # (B, H, W, C)

        # Projection and dropout
        x_out = proj(x_out)
        x_out = proj_drop_layer(x_out)

        return x_out


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "100_WeightedPermuteMLP.json")
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

        dim = next(inp for inp in inputs if inp["name"] == "dim")["value"]
        seg_dim = next(inp for inp in inputs if inp["name"] == "seg_dim")["value"]
        qkv_bias = next((inp["value"] for inp in inputs if inp["name"] == "qkv_bias"), False)
        proj_drop = next((inp["value"] for inp in inputs if inp["name"] == "proj_drop"), 0.0)

        input_groups.append([x, dim, seg_dim, qkv_bias, proj_drop])
    return input_groups


def get_init_inputs():
    return []