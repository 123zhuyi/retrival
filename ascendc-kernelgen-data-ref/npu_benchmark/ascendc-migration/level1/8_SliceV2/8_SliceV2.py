import torch
import torch.nn as nn
import torch_npu
import json
import os


class Model(nn.Module):
    r"""
    torch_npu.npu_slice(self, offsets, size) -> Tensor

    Extracts a slice from the input tensor.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def npu_slice_native(self, offsets, size):
            slices = []
            for i in range(self.ndim):
                start = offsets[i]
                end = offsets[i] + size[i]
                slices.append(slice(start, end))
            return self[tuple(slices)]

    Args:
        self (Tensor): Input tensor. Supports fp32, fp16, bf16, fp64,
            int8, int16, int32, int64, uint8, bool.
        offsets (List[int]): Starting indices for each dimension.
        size (List[int]): Size of the slice in each dimension.

    Returns:
        Tensor: The sliced tensor.

    Example::

        >>> input = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]], dtype=torch.float16)
        >>> offsets = [0, 0]
        >>> size = [2, 2]
        >>> torch_npu.npu_slice(input, offsets, size)
        tensor([[1., 2.],
                [6., 7.]], dtype=torch.float16)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, offsets: list, size: list) -> torch.Tensor:
        """
        Extracts a slice from the input tensor.

        Args:
            input (torch.Tensor): Input tensor.
            offsets (list): Starting indices for each dimension.
            size (list): Size of the slice in each dimension.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        return torch_npu.npu_slice(input, offsets, size)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "8_SliceV2.json")

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
        offsets_info = inputs[1]
        size_info = inputs[2]

        input_dtype = dtype_map[input_info["dtype"]]
        input_shape = input_info["shape"]
        input_fill = input_info.get("fill", None)

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

        offsets = offsets_info["value"]
        size = size_info["value"]

        input_groups.append([input_tensor, offsets, size])

    return input_groups


def get_init_inputs():
    return []