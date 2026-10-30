import torch
import torch.nn as nn
import json
import os
import numpy as np


class Model(nn.Module):
    """
    Simple model that performs cumulative sum along a specified dimension.
    torch.cumsum = cumsum(...)
    cumsum(input, dim, *, dtype=None, out=None) -> Tensor

    Returns the cumulative sum of elements of :attr:`input` in the dimension
    :attr:`dim`.

    For example, if :attr:`input` is a vector of size N, the result will also be
    a vector of size N, with elements.

    .. math::
        y_i = x_1 + x_2 + x_3 + \dots + x_i

    Args:
        input (Tensor): the input tensor.
        dim  (int): the dimension to do the operation over

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.
        out (Tensor, optional): the output tensor.

    Example::

        >>> a = torch.randint(1, 20, (10,))
        >>> a
        tensor([13,  7,  3, 10, 13,  3, 15, 10,  9, 10])
        >>> torch.cumsum(a, dim=0)
        tensor([13, 20, 23, 33, 46, 49, 64, 74, 83, 93])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Applies cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int): The dimension to do the operation over.

        Returns:
            torch.Tensor: Output tensor with cumulative sum, same shape as input.
        """
        return torch.cumsum(x, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_Cumsum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]

        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5)
        dim = dim_info["value"]
        input_groups.append([x, dim])
    return input_groups


def get_init_inputs():
    return []
