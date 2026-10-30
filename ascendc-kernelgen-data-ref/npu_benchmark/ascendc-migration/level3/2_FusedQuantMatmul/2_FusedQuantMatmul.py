import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    aclnnFusedQuantMatmul

    Fused quantized matrix multiplication with dequantization, bias addition, and GELU activation.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def fused_quant_matmul_native(x1, x2, x1scale, x2scale, bias, fused_op_type, out_dtype, transpose_x1, transpose_x2):
            # 1. Optional transpose
            if transpose_x1:
                x1 = x1.transpose(-2, -1)
            if transpose_x2:
                x2 = x2.transpose(-2, -1)

            # 2. MatMul in float32 for precision
            temp = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))

            # 3. Dequantization and bias handling
            x2scale_f = x2scale.to(torch.float32)
            x1scale_f = x1scale.unsqueeze(-1).to(torch.float32)

            if bias is None:
                out = temp * x2scale_f * x1scale_f
            elif bias.dtype == torch.int32:
                out = (temp + bias.to(torch.float32)) * x2scale_f * x1scale_f
            else:
                out = temp * x2scale_f * x1scale_f + bias.to(torch.float32)

            # 4. GELU
            if fused_op_type == "gelu_erf":
                out = torch.nn.functional.gelu(out, approximate='none')
            elif fused_op_type == "gelu_tanh":
                out = torch.nn.functional.gelu(out, approximate='tanh')

            # 5. Cast to output dtype
            return out.to(out_dtype)

    Args:
        x1 (Tensor): Quantized activation tensor.
        x2 (Tensor): Quantized weight tensor.
        x1scale (Tensor): Per-token dequantization scale.
        x2scale (Tensor): Per-channel dequantization scale.
        bias (Tensor, optional): Bias tensor. Can be int32 or fp16/bf16/fp32.
        fused_op_type (str): Post-processing type, "gelu_erf" or "gelu_tanh".
        out_dtype (torch.dtype): Output data type.
        transpose_x1 (bool, optional): Whether to transpose x1. Default: False.
        transpose_x2 (bool, optional): Whether to transpose x2. Default: False.

    Returns:
        Tensor: Output tensor after fused quant matmul + dequant + bias + gelu.

    Example::

        >>> x1 = torch.randint(-8, 8, (2, 4), dtype=torch.int8)
        >>> x2 = torch.randint(-8, 8, (4, 8), dtype=torch.int8)
        >>> x1scale = torch.tensor([0.5, 0.5], dtype=torch.float32)
        >>> x2scale = torch.ones(8, dtype=torch.float32)
        >>> bias = torch.randn(8, dtype=torch.float16)
        >>> fused_quant_matmul_native(x1, x2, x1scale, x2scale, bias, "gelu_erf", torch.float16)
        tensor([...], dtype=torch.float16)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x1scale: torch.Tensor,
                x2scale: torch.Tensor, bias, fused_op_type: str, out_dtype,
                transpose_x1: bool = False, transpose_x2: bool = False) -> torch.Tensor:
        """
        Fused quantized matrix multiplication with dequantization, bias, and GELU.

        Args:
            x1 (torch.Tensor): Quantized activation.
            x2 (torch.Tensor): Quantized weight.
            x1scale (torch.Tensor): Per-token scale.
            x2scale (Tensor): Per-channel scale.
            bias (torch.Tensor, optional): Bias.
            fused_op_type (str): "gelu_erf" or "gelu_tanh".
            out_dtype (torch.dtype): Output dtype.
            transpose_x1 (bool, optional): Transpose x1. Default: False.
            transpose_x2 (bool, optional): Transpose x2. Default: False.

        Returns:
            torch.Tensor: Output tensor.
        """
        # 1. Optional transpose
        if transpose_x1:
            x1 = x1.transpose(-2, -1)
        if transpose_x2:
            x2 = x2.transpose(-2, -1)

        # 2. MatMul in float32 for precision
        temp = torch.matmul(x1.to(torch.float32), x2.to(torch.float32))

        # 3. Dequantization and bias handling
        x2scale_f = x2scale.to(torch.float32)
        x1scale_f = x1scale.unsqueeze(-1).to(torch.float32)

        if bias is None:
            out = temp * x2scale_f * x1scale_f
        elif bias.dtype == torch.int32:
            out = (temp + bias.to(torch.float32)) * x2scale_f * x1scale_f
        else:
            out = temp * x2scale_f * x1scale_f + bias.to(torch.float32)

        # 4. GELU
        if fused_op_type == "gelu_erf":
            out = torch.nn.functional.gelu(out, approximate='none')
        elif fused_op_type == "gelu_tanh":
            out = torch.nn.functional.gelu(out, approximate='tanh')

        # 5. Cast to output dtype
        return out.to(out_dtype)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "2_FusedQuantMatmul.json")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int8": torch.int8,
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

        x1_info = inputs[0]
        x2_info = inputs[1]
        x1scale_info = inputs[2]
        x2scale_info = inputs[3]
        bias_info = inputs[4] if len(inputs) > 4 else {"value": None}
        fused_op_type_info = inputs[5] if len(inputs) > 5 else {"value": "gelu_erf"}
        out_dtype_info = inputs[6] if len(inputs) > 6 else {"value": "float16"}
        transpose_x1_info = inputs[7] if len(inputs) > 7 else {"value": False}
        transpose_x2_info = inputs[8] if len(inputs) > 8 else {"value": False}

        def make_tensor(info, idx):
            dtype = dtype_map[info["dtype"]]
            shape = info["shape"]
            fill = info.get("fill", None)
            if fill in fill_map:
                return torch.full(shape, fill_map[fill], dtype=dtype)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, shape, dtype=torch.float32)
                return raw.to(dtype)
            else:
                raw = torch.empty(shape, dtype=torch.float32).uniform_(-5, 5)
                return raw.to(dtype)

        x1 = make_tensor(x1_info, idx)
        x2 = make_tensor(x2_info, idx)
        x1scale = make_tensor(x1scale_info, idx)
        x2scale = make_tensor(x2scale_info, idx)

        # bias: None if value is None, otherwise generate tensor
        bias = None
        if bias_info.get("value") is not None:
            bias = make_tensor(bias_info, idx)
        elif "dtype" in bias_info:
            bias = make_tensor(bias_info, idx)

        fused_op_type = fused_op_type_info["value"]
        out_dtype_str = out_dtype_info["value"]
        out_dtype = dtype_map.get(out_dtype_str, torch.float16)

        transpose_x1 = transpose_x1_info["value"]
        transpose_x2 = transpose_x2_info["value"]

        input_groups.append([
            x1, x2, x1scale, x2scale, bias,
            fused_op_type, out_dtype, transpose_x1, transpose_x2
        ])

    return input_groups


def get_init_inputs():
    return []