import torch
import torch.nn as nn
import json
import os
import torch_npu

class Model(nn.Module):
    r"""
    torch.scatter_add = scatter_add(input, dim, index, src) -> Tensor

    Adds all values from the tensor src into input at the indices specified in
    the index tensor. For each value in src, its output index is specified by
    its index in src for dimension != dim and by the corresponding value in
    index for dimension = dim.

    Equivalent to aclnnScatterAdd.

    Args:
        input (Tensor): the input tensor.
        dim (int): the axis along which to index.
        index (LongTensor): the indices of elements to scatter.
        src (Tensor): the source tensor.

    Returns:
        Tensor: the output tensor.

    Example::

        >>> input = torch.zeros(3, 5)
        >>> src = torch.ones(2, 5)
        >>> index = torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 1, 2]])
        >>> torch.scatter_add(input, 1, index, src)
        tensor([[2., 1., 1., 0., 0.],
                [1., 2., 1., 1., 1.],
                [0., 0., 0., 0., 0.]])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, dim: int, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
        Scatter-adds src into input at indices.

        Args:
            input (torch.Tensor): Input tensor.
            dim (int): Dimension along which to index.
            index (torch.Tensor): Indices tensor.
            src (torch.Tensor): Source tensor.

        Returns:
            torch.Tensor: Result tensor.
        """
        return torch.scatter_add(input, dim, index, src)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "21_Scatter.json")

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
        dim_info = inputs[1]
        index_info = inputs[2]
        src_info = inputs[3]

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

        dim = dim_info["value"]

        # index: use range from json if provided
        index_range = index_info.get("range", [0, x_shape[dim] - 1])
        index_shape = tuple(index_info["shape"])
        index = torch.randint(index_range[0], index_range[1] + 1, index_shape, dtype=torch.int64)

        src_dtype = dtype_map[src_info["dtype"]]
        src_shape = src_info["shape"]
        src_fill = src_info.get("fill", None)

        if src_fill in fill_map:
            src = torch.full(src_shape, fill_map[src_fill], dtype=src_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, src_shape, dtype=torch.float32)
            src = raw.to(src_dtype)
        else:
            raw = torch.empty(src_shape, dtype=torch.float32).uniform_(-5, 5)
            src = raw.to(src_dtype)

        input_groups.append([x, dim, index, src])

    return input_groups


def get_init_inputs():
    return []