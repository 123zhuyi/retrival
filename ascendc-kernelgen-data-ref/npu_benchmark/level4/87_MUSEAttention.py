import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import math


class Model(nn.Module):
    """
    MUSE (Multi-Scale) Attention: combines standard multi-head scaled dot-product
    self-attention with a multi-scale convolutional branch using depthwise separable
    convolutions of kernel sizes 1, 3, and 5, weighted by learned dynamic parameters.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                d_model: int, d_k: int, d_v: int, h: int) -> torch.Tensor:
        """
        Args:
            queries: (batch, nq, d_model)
            keys:    (batch, nk, d_model)
            values:  (batch, nk, d_model)
            d_model: model dimension
            d_k:     query/key dimension per head
            d_v:     value dimension per head
            h:       number of heads

        Returns:
            output: (batch, nq, d_model)
        """
        torch.manual_seed(42)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # Business key for initialization (no device/dtype)
        business_key = (d_model, d_k, d_v, h)
        cache_key = (business_key, queries.device, queries.dtype)
        if cache_key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(business_key) & 0xFFFFFFFF)

            # FC layers for Q, K, V, and output
            fc_q = nn.Linear(d_model, h * d_k).to(device=queries.device, dtype=queries.dtype)
            fc_k = nn.Linear(d_model, h * d_k).to(device=queries.device, dtype=queries.dtype)
            fc_v = nn.Linear(d_model, h * d_v).to(device=queries.device, dtype=queries.dtype)
            fc_o = nn.Linear(h * d_v, d_model).to(device=queries.device, dtype=queries.dtype)

            # Depthwise separable convs for kernel sizes 1, 3, 5
            in_ch = h * d_v
            out_ch = d_model
            # conv1: kernel 1 => depthwise is Identity (no parameters, no to needed but we put in cache)
            conv1 = nn.Identity()
            conv1_pointwise = nn.Conv1d(in_ch, out_ch, 1).to(device=queries.device, dtype=queries.dtype)
            # conv3
            conv3_depth = nn.Conv1d(in_ch, in_ch, 3, groups=in_ch, padding=1).to(device=queries.device, dtype=queries.dtype)
            conv3_pointwise = nn.Conv1d(in_ch, out_ch, 1).to(device=queries.device, dtype=queries.dtype)
            # conv5
            conv5_depth = nn.Conv1d(in_ch, in_ch, 5, groups=in_ch, padding=2).to(device=queries.device, dtype=queries.dtype)
            conv5_pointwise = nn.Conv1d(in_ch, out_ch, 1).to(device=queries.device, dtype=queries.dtype)

            # Dynamic parameters
            dy_paras = nn.Parameter(torch.ones(3, device=queries.device, dtype=queries.dtype))

            self._cache[cache_key] = (fc_q, fc_k, fc_v, fc_o,
                                      conv1, conv1_pointwise,
                                      conv3_depth, conv3_pointwise,
                                      conv5_depth, conv5_pointwise,
                                      dy_paras)
            torch.set_rng_state(rng_state)

        (fc_q, fc_k, fc_v, fc_o,
         conv1, conv1_pointwise,
         conv3_depth, conv3_pointwise,
         conv5_depth, conv5_pointwise,
         dy_paras) = self._cache[cache_key]

        # ---------- Self-attention branch ----------
        q = fc_q(queries).view(b_s, nq, h, d_k).permute(0, 2, 1, 3)          # (b_s, h, nq, d_k)
        k = fc_k(keys).view(b_s, nk, h, d_k).permute(0, 2, 3, 1)            # (b_s, h, d_k, nk)
        v = fc_v(values).view(b_s, nk, h, d_v).permute(0, 2, 1, 3)          # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / math.sqrt(d_k)                           # (b_s, h, nq, nk)
        att = F.softmax(att, dim=-1)
        # dropout is 0, so we skip it

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, h * d_v)  # (b_s, nq, h*d_v)
        out = fc_o(out)  # (b_s, nq, d_model)

        # ---------- Multi-scale convolutional branch ----------
        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)           # (bs, h*d_v, nk)
        # Apply depthwise then pointwise for each kernel
        def apply_conv(depth_conv, pointwise_conv, x):
            if isinstance(depth_conv, nn.Identity):
                mid = x
            else:
                mid = depth_conv(x)
            return pointwise_conv(mid)

        conv1_out = apply_conv(conv1, conv1_pointwise, v2)
        conv3_out = apply_conv(conv3_depth, conv3_pointwise, v2)
        conv5_out = apply_conv(conv5_depth, conv5_pointwise, v2)

        # Softmax over dy_paras and weighted sum
        w = F.softmax(dy_paras, dim=0)  # (3,)
        out2 = w[0] * conv1_out + w[1] * conv3_out + w[2] * conv5_out       # (bs, d_model, nk)

        # Align out2 length to nq via adaptive pooling over the sequence dimension
        if nk != nq:
            out2 = F.adaptive_avg_pool1d(out2, nq)                          # (bs, d_model, nq)
        out2 = out2.permute(0, 2, 1)                                        # (bs, nq, d_model)

        out = out + out2
        return out


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "87_MUSEAttention.json")
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
        q_info = next(inp for inp in inputs if inp["name"] == "queries")
        k_info = next(inp for inp in inputs if inp["name"] == "keys")
        v_info = next(inp for inp in inputs if inp["name"] == "values")
        dtype = dtype_map[q_info["dtype"]]

        queries = random_tensor(q_info["shape"], dtype)
        keys = random_tensor(k_info["shape"], dtype)
        values = random_tensor(v_info["shape"], dtype)

        d_model = next(inp for inp in inputs if inp["name"] == "d_model")["value"]
        d_k = next(inp for inp in inputs if inp["name"] == "d_k")["value"]
        d_v = next(inp for inp in inputs if inp["name"] == "d_v")["value"]
        h = next(inp for inp in inputs if inp["name"] == "h")["value"]

        input_groups.append([queries, keys, values, d_model, d_k, d_v, h])
    return input_groups


def get_init_inputs():
    return []