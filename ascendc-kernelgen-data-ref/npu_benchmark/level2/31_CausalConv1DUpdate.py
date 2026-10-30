import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Model that performs causal 1D convolution update.

    This operator concatenates the convolution state history with the current input,
    updates the state in-place, and applies a grouped 1D convolution.

    PyTorch native implementation of forward function:
    def forward(self, x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor = None, activation: str = None, padding: int = 0):
        # 1. padding 处理（对齐 ATK TorchCpuApi）
        batch_size = x.shape[0] - padding
        x = x[:batch_size]
        torch.manual_seed(1234)
        conv_state_indices = torch.randperm(conv_state.shape[0])[:batch_size].to(dtype=torch.int32, device=x.device)
        conv_state = conv_state[conv_state_indices].detach().clone()

        # 2. 原有计算逻辑
        dtype_in = x.dtype
        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)

        batch, dim, seqlen = x.shape
        width = weight.shape[1]
        state_len = conv_state.shape[-1]

        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])

        left_part = x_new[:, :, :width - 1]
        right_part = x_new[:, :, -seqlen:]
        concat_input = torch.cat([left_part, right_part], dim=2)

        out = F.conv1d(concat_input, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]

        if unsqueeze:
            out = out.squeeze(-1)
        if activation == "silu":
            out = F.silu(out)

        return out.to(dtype_in)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor = None, activation: str = None, padding: int = 0):
        """
        Performs causal 1D convolution update.

        Args:
            x (Tensor): Input tensor, shape (batch, dim) or (batch, dim, seqlen).
            conv_state (Tensor): Convolution state cache pool, shape (num_cache_lines, dim, state_len).
            weight (Tensor): Convolution weight, shape (dim, width).
            bias (Tensor, optional): Bias term, shape (dim,).
            activation (str, optional): Activation function, "silu" or None.
            padding (int): Number of padded slots in x's batch dim. Effective batch = x.shape[0] - padding.

        Returns:
            Tensor: Output tensor, shape (effective_batch, ...) same as x[:effective_batch].
        """
        # 1. padding 处理（对齐 ATK TorchCpuApi）
        batch_size = x.shape[0] - padding
        x = x[:batch_size]
        torch.manual_seed(1234)
        conv_state_indices = torch.randperm(conv_state.shape[0])[:batch_size].to(dtype=torch.int32, device=x.device)
        conv_state = conv_state[conv_state_indices].detach().clone()

        # 2. 原有计算逻辑
        dtype_in = x.dtype
        unsqueeze = x.dim() == 2
        if unsqueeze:
            x = x.unsqueeze(-1)

        batch, dim, seqlen = x.shape
        width = weight.shape[1]
        state_len = conv_state.shape[-1]

        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])

        left_part = x_new[:, :, :width - 1]
        right_part = x_new[:, :, -seqlen:]
        concat_input = torch.cat([left_part, right_part], dim=2)

        out = F.conv1d(concat_input, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]

        if unsqueeze:
            out = out.squeeze(-1)
        if activation == "silu":
            out = F.silu(out)

        return out.to(dtype_in)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "CausalConv1dUpdate.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        """独立随机选择正态/均匀分布"""
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-100.0, 100.0).item())
            sigma = float(torch.empty(1).uniform_(1.0, 25.0).item())
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
            "bf16": torch.bfloat16,
        }

        tensors = {}
        for inp in inputs:
            name = inp.get("name", "")
            dtype_str = inp.get("dtype", "float16")
            dtype = dtype_map.get(dtype_str, torch.float16)

            if inp["type"] == "tensor":
                shape = inp["shape"]
                tensors[name] = random_tensor(shape, dtype)
            elif inp["type"] == "attr":
                # attr 值在 range_values 字段
                tensors[name] = inp.get("range_values")

        # 按 forward 签名顺序组装
        x = tensors["x"]
        conv_state = tensors["conv_state"]
        weight = tensors["weight"]
        bias = tensors.get("bias") if tensors.get("with_bias", False) else None
        activation = "silu" if tensors.get("activation", False) else None
        padding = tensors.get("padding", 0)

        input_groups.append([x, conv_state, weight, bias, activation, padding])
    return input_groups


def get_init_inputs():
    return []