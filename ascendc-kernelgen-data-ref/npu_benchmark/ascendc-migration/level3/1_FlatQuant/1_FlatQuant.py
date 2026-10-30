import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    aclnnFlatQuant = FlatQuant(x, kroneckerP1, kroneckerP2, clipRatio) -> (Tensor, Tensor)

    Fused Kronecker transformation and per-channel INT4 quantization.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def flat_quant_native(x, kronecker_p1, kronecker_p2, clip_ratio=1.0):
            # 1. Kronecker transformation
            x1 = x @ kronecker_p2
            x2 = kronecker_p1.to(torch.float32) @ x1.to(torch.float32)
            x2 = x2.to(torch.float16)
            x2 = x2.flatten(-2, -1)

            # 2. Per-channel absmax quantization
            qscale = torch.abs(x2).max(dim=-1, keepdim=True)[0].to(torch.float32)
            ratio = 7.0 / clip_ratio
            qscale2 = ratio / qscale
            out = x2 * qscale2.to(torch.float16)

            # 3. Clamp, round and cast to int8
            out = torch.clamp(out, -8, 7)
            out = torch.round(out).to(torch.int8).reshape(x.shape)

            # 4. Compute quantization scale
            quant_scale = torch.flatten(qscale / ratio)

            return out, quant_scale

    Args:
        x (Tensor): Input tensor. Supports fp16, fp32, bf16.
        kronecker_p1 (Tensor): First Kronecker factor matrix.
        kronecker_p2 (Tensor): Second Kronecker factor matrix.
        clip_ratio (float, optional): Clipping ratio for quantization range. Default: 1.0.

    Returns:
        tuple: (quantized_output, quant_scale)
            quantized_output: INT8 tensor (unpacked INT4 values in INT8 container).
            quant_scale: Per-channel quantization scale.

    Example::

        >>> x = torch.randn(2, 4, 8, dtype=torch.float16)
        >>> p1 = torch.randn(4, 4, dtype=torch.float16)
        >>> p2 = torch.randn(8, 8, dtype=torch.float16)
        >>> out, scale = flat_quant_native(x, p1, p2, clip_ratio=1.0)
        >>> out.dtype
        torch.int8
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kronecker_p1: torch.Tensor,
                kronecker_p2: torch.Tensor, clip_ratio: float = 1.0) -> tuple:
        """
        Fused Kronecker transformation and INT4 quantization.

        Args:
            x (torch.Tensor): Input tensor.
            kronecker_p1 (torch.Tensor): First Kronecker factor.
            kronecker_p2 (torch.Tensor): Second Kronecker factor.
            clip_ratio (float, optional): Clipping ratio. Default: 1.0.

        Returns:
            tuple: (quantized_output, quant_scale).
        """
        x_shape = x.shape

        # 1. Kronecker transformation
        x1 = x @ kronecker_p2
        x2 = kronecker_p1.to(torch.float32) @ x1.to(torch.float32)
        x2 = x2.to(torch.float16)
        x2 = x2.flatten(-2, -1)

        # 2. Per-channel absmax quantization
        qscale = torch.abs(x2).max(dim=-1, keepdim=True)[0].to(torch.float32)
        ratio = 7.0 / clip_ratio
        qscale2 = ratio / qscale
        out = x2 * qscale2.to(torch.float16)

        # 3. Clamp, round and cast to int8
        out = torch.clamp(out, -8, 7)
        out = torch.round(out).to(torch.int8).reshape(x_shape)

        # 4. Compute quantization scale
        quant_scale = torch.flatten(qscale / ratio)

        return out, quant_scale


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "1_FlatQuant.json")

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
        p1_info = inputs[1]
        p2_info = inputs[2]
        clip_ratio_info = inputs[3] if len(inputs) > 3 else {"value": 1.0}

        x_dtype = dtype_map[x_info["dtype"]]
        x_shape = x_info["shape"]
        x_fill = x_info.get("fill", None)

        if x_fill in fill_map:
            x = torch.full(x_shape, fill_map[x_fill], dtype=x_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, x_shape, dtype=torch.float32)
            x = raw.to(x_dtype)
        else:
            raw = torch.empty(x_shape, dtype=torch.float32).uniform_(-5, 5)
            x = raw.to(x_dtype)

        p1_dtype = dtype_map[p1_info["dtype"]]
        p1_shape = p1_info["shape"]
        p1_fill = p1_info.get("fill", None)

        if p1_fill in fill_map:
            p1 = torch.full(p1_shape, fill_map[p1_fill], dtype=p1_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, p1_shape, dtype=torch.float32)
            p1 = raw.to(p1_dtype)
        else:
            raw = torch.empty(p1_shape, dtype=torch.float32).uniform_(-5, 5)
            p1 = raw.to(p1_dtype)

        p2_dtype = dtype_map[p2_info["dtype"]]
        p2_shape = p2_info["shape"]
        p2_fill = p2_info.get("fill", None)

        if p2_fill in fill_map:
            p2 = torch.full(p2_shape, fill_map[p2_fill], dtype=p2_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, p2_shape, dtype=torch.float32)
            p2 = raw.to(p2_dtype)
        else:
            raw = torch.empty(p2_shape, dtype=torch.float32).uniform_(-5, 5)
            p2 = raw.to(p2_dtype)

        clip_ratio = clip_ratio_info["value"]

        input_groups.append([x, p1, p2, clip_ratio])

    return input_groups


def get_init_inputs():
    return []