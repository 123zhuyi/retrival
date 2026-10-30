import torch
import torch_npu
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Simple model that performs Group Normalization with SiLU activation.
    torch_npu.npu_group_norm_silu(input, weight, bias, group, eps=1e-5) -> (Tensor out, Tensor meanOut, Tensor rstdOut)

    Pure PyTorch implementation (replacing torch_npu.npu_group_norm_silu):

    def forward(self, input, weight, bias, group, eps=1e-5):
        import torch.nn.functional as F
        
        N, C = input.shape[0], input.shape[1]
        # Reshape to [N, group, C//group, ...]
        input_reshaped = input.view(N, group, C // group, *input.shape[2:])
        
        # Compute mean and var over spatial dims
        dims = list(range(3, input_reshaped.ndim))
        mean = input_reshaped.mean(dim=dims, keepdim=True)
        var = input_reshaped.var(dim=dims, unbiased=False, keepdim=True)
        
        # Normalize
        x_norm = (input_reshaped - mean) / torch.sqrt(var + eps)
        x_norm = x_norm.view(input.shape)
        
        # Apply weight and bias (affine)
        if weight is not None:
            weight = weight.view(1, -1, *([1] * (input.ndim - 2)))
            x_norm = x_norm * weight
        if bias is not None:
            bias = bias.view(1, -1, *([1] * (input.ndim - 2)))
            x_norm = x_norm + bias
        
        # SiLU activation: x * sigmoid(x)
        output = F.silu(x_norm)
        # Or manually: output = x_norm * torch.sigmoid(x_norm)
        
        # Compute mean and rstd for return
        mean_out = mean.view(N, group)
        rstd = 1.0 / torch.sqrt(var + eps)
        rstd_out = rstd.view(N, group)
        
        return output, mean_out, rstd_out
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, weight, bias, group, eps):
        return torch_npu.npu_group_norm_silu(input, weight, bias, group, eps)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "5_GroupNormSilu.json")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    fill_map = {
        'inf': float('inf'),
        '-inf': float('-inf'),
        'nan': float('nan'),
        '-nan': -float('nan'),
        'zero': 0.0,
    }

    input_groups = []

    for idx, case in enumerate(cases):
        inputs = case["inputs"]

        input_info = inputs[0]
        weight_info = inputs[1]
        bias_info = inputs[2]
        group_info = inputs[3]
        eps_info = inputs[4]

        input_dtype = dtype_map[input_info["dtype"]]
        input_shape = input_info["shape"]
        input_fill = input_info.get("fill", None)

        group = group_info["value"]
        eps = eps_info["value"]

        if input_fill in fill_map:
            input_tensor = torch.full(input_shape, fill_map[input_fill], dtype=input_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, input_shape, dtype=torch.float32)
            input_tensor = raw.to(input_dtype)
        else:
            raw = torch.empty(input_shape, dtype=torch.float32).uniform_(-5, 5)
            input_tensor = raw.to(input_dtype)

        if weight_info.get("shape") is not None:
            weight_dtype = dtype_map[weight_info["dtype"]]
            weight_shape = weight_info["shape"]
            weight_fill = weight_info.get("fill", None)
            if weight_fill in fill_map:
                weight_tensor = torch.full(weight_shape, fill_map[weight_fill], dtype=weight_dtype)
            elif idx % 2 == 0:
                weight_tensor = torch.randn(weight_shape, dtype=weight_dtype)
            else:
                weight_tensor = torch.empty(weight_shape, dtype=weight_dtype).uniform_(-1, 1)
        else:
            weight_tensor = None

        if bias_info.get("shape") is not None:
            bias_dtype = dtype_map[bias_info["dtype"]]
            bias_shape = bias_info["shape"]
            bias_fill = bias_info.get("fill", None)
            if bias_fill in fill_map:
                bias_tensor = torch.full(bias_shape, fill_map[bias_fill], dtype=bias_dtype)
            elif idx % 2 == 0:
                bias_tensor = torch.randn(bias_shape, dtype=bias_dtype)
            else:
                bias_tensor = torch.empty(bias_shape, dtype=bias_dtype).uniform_(-1, 1)
        else:
            bias_tensor = None

        input_groups.append([input_tensor, weight_tensor, bias_tensor, group, eps])

    return input_groups


def get_init_inputs():
    return []