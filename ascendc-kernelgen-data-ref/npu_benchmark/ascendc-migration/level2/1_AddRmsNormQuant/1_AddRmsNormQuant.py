import torch
import torch.nn as nn
import torch_npu
import json
import os


class Model(nn.Module):
    r"""
    torch_npu.npu_add_rms_norm_quant = npu_add_rms_norm_quant(x, residual, gamma, scale, offset, epsilon=1e-6) -> Tensor

    Fused operator that performs Add, RMSNorm, and Quantize in a single kernel.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def add_rms_norm_quant_native(x, residual, gamma, scale, offset, eps=1e-6):
            # Step 1: Add residual
            added = x + residual
            # Step 2: RMSNorm
            variance = added.pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps)
            normed = added * rstd * gamma
            # Step 3: Quantize
            quantized = torch.quantize_per_tensor(normed, scale, offset, torch.qint8)
            return quantized, normed, rstd.squeeze(-1)

    Args:
        x (Tensor): Input tensor. Supports fp32, fp16, bf16.
        residual (Tensor): Residual tensor to add, same shape as x.
        gamma (Tensor): RMSNorm weight tensor, 1D, size equal to last dim of x.
        scale (float): Quantization scale (inverse scale passed to kernel).
        offset (int): Quantization zero-point offset.
        epsilon (float, optional): Value added to denominator for numerical stability. Default: 1e-6.

    Returns:
        tuple: (quantized_output, normed_output, rstd)

    Example::

        >>> x = torch.randn(2, 4, dtype=torch.float16)
        >>> residual = torch.randn(2, 4, dtype=torch.float16)
        >>> gamma = torch.ones(4, dtype=torch.float16)
        >>> scale = 1.0
        >>> offset = 0
        >>> torch_npu.npu_add_rms_norm_quant(x, residual, gamma, scale, offset, epsilon=1e-6)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor,
                scale: float, offset: int, epsilon: float = 1e-6) -> tuple:
        """
        Fused Add + RMSNorm + Quantize.

        Args:
            x (torch.Tensor): Input tensor.
            residual (torch.Tensor): Residual tensor.
            gamma (torch.Tensor): RMSNorm weight.
            scale (float): Quantization scale.
            offset (int): Quantization zero-point.
            epsilon (float, optional): RMSNorm epsilon. Default: 1e-6.

        Returns:
            tuple: (quantized_output, normed_output, rstd)
        """
        return torch_npu.npu_add_rms_norm_quant(x, residual, gamma, scale, offset, epsilon=epsilon)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "1_AddRmsNormQuant.json")

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
        scene = case.get("scene", "")

        x_info = inputs[0]
        residual_info = inputs[1]
        gamma_info = inputs[2]
        scale_info = inputs[3]
        offset_info = inputs[4]
        eps_info = inputs[5] if len(inputs) > 5 else {"value": 1e-6}

        dtype = dtype_map[x_info["dtype"]]
        x_shape = x_info["shape"]
        x_fill = x_info.get("fill", None)

        if x_fill in fill_map:
            x = torch.full(x_shape, fill_map[x_fill], dtype=dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, x_shape, dtype=torch.float32)
            x = raw.to(dtype)
        else:
            raw = torch.empty(x_shape, dtype=torch.float32).uniform_(-5, 5)
            x = raw.to(dtype)

        residual_shape = residual_info["shape"]
        residual_fill = residual_info.get("fill", None)
        if residual_fill in fill_map:
            residual = torch.full(residual_shape, fill_map[residual_fill], dtype=dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, residual_shape, dtype=torch.float32)
            residual = raw.to(dtype)
        else:
            raw = torch.empty(residual_shape, dtype=torch.float32).uniform_(-5, 5)
            residual = raw.to(dtype)

        gamma = torch.ones(gamma_info["shape"], dtype=dtype)
        scale = scale_info["value"]
        offset = offset_info["value"]
        eps = eps_info["value"]

        input_groups.append([x, residual, gamma, scale, offset, eps])

    return input_groups


def get_init_inputs():
    return []