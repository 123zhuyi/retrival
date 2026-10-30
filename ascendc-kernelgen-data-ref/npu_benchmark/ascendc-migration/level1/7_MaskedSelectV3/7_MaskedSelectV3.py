import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    torch.masked_select = masked_select(input, mask, out=None) -> Tensor

    Returns a new 1-D tensor which indexes the input tensor according to
    the boolean mask which is a BoolTensor.

    The shapes of the mask tensor and the input tensor don't need to match,
    but they must be broadcastable.

    .. math::
        out = \left[ \text{input}[i] \right]_{i \in \mathcal{I}}, \quad \text{where } \mathcal{I} = \left\{ i \mid \text{mask}[i] = \text{True} \right\}

    Args:
        input (Tensor): The input tensor, supports 1-8D tensors.
                        Supports fp32, fp16, bf16, fp64, int8, int16, int32,
                        int64, uint8, bool.
        mask (Tensor): The boolean mask tensor, supports 1-8D tensors.
                        Must be broadcastable with input. Dtype must be bool.

    Returns:
        Tensor: A 1-D tensor of shape (n,), where n is the number of True
                elements in the mask.

    Example::

        >>> x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        >>> mask = torch.tensor([False, False, False, False, True, True, True, True])
        >>> torch.masked_select(x, mask)
        tensor([4, 5, 6, 7])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies masked select to the input tensor according to the boolean mask.

        Args:
            input (torch.Tensor): Input tensor to be selected from.
            mask (torch.Tensor): Boolean mask tensor for selection.

        Returns:
            torch.Tensor: 1-D tensor containing selected elements.
        """
        return torch.masked_select(input, mask)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "7_MaskedSelectV3.json")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
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

        input_info = inputs[0]
        mask_info = inputs[1]

        input_dtype = dtype_map[input_info["dtype"]]
        input_shape = input_info["shape"]
        input_fill = input_info.get("fill", None)

        mask_dtype = dtype_map[mask_info["dtype"]]
        mask_shape = mask_info["shape"]
        mask_fill = mask_info.get("fill", None)

        # Generate input tensor (self)
        if input_fill in fill_map:
            fill_val = fill_map[input_fill]
            if input_dtype == torch.bool:
                input_tensor = torch.full(input_shape, fill_val != 0.0, dtype=input_dtype)
            elif input_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                if input_fill in ('inf', '-inf', 'nan', '-nan'):
                    input_tensor = torch.zeros(input_shape, dtype=input_dtype)
                else:
                    input_tensor = torch.full(input_shape, int(fill_val), dtype=input_dtype)
            else:
                input_tensor = torch.full(input_shape, fill_val, dtype=input_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            if input_dtype == torch.bool:
                raw = torch.normal(mu, sigma, input_shape, dtype=torch.float32)
                input_tensor = raw > 0
            elif input_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raw = torch.normal(mu, sigma, input_shape, dtype=torch.float32)
                input_tensor = raw.to(input_dtype)
            else:
                raw = torch.normal(mu, sigma, input_shape, dtype=torch.float32)
                input_tensor = raw.to(input_dtype)
        else:
            if input_dtype == torch.bool:
                raw = torch.empty(input_shape, dtype=torch.float32).uniform_(-5, 5)
                input_tensor = raw > 0
            elif input_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raw = torch.empty(input_shape, dtype=torch.float32).uniform_(-5, 5)
                input_tensor = raw.to(input_dtype)
            else:
                raw = torch.empty(input_shape, dtype=torch.float32).uniform_(-5, 5)
                input_tensor = raw.to(input_dtype)

        # Generate mask tensor
        if mask_fill in fill_map:
            fill_val = fill_map[mask_fill]
            mask_tensor = torch.full(mask_shape, fill_val != 0.0, dtype=mask_dtype)
        elif mask_fill in ('true', 'True', '1', 1, True):
            mask_tensor = torch.full(mask_shape, True, dtype=mask_dtype)
        elif mask_fill in ('false', 'False', '0', 0, False):
            mask_tensor = torch.full(mask_shape, False, dtype=mask_dtype)
        elif idx % 2 == 0:
            mask_tensor = torch.empty(mask_shape, dtype=torch.float32).uniform_(-1, 1) > 0
        else:
            mask_tensor = torch.empty(mask_shape, dtype=torch.float32).uniform_(-1, 1) > 0

        input_groups.append([input_tensor, mask_tensor])

    return input_groups


def get_init_inputs():
    return []