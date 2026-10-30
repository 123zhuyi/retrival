import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that sorts a tensor along a specified dimension.
    torch.sort = sort(...)
    sort(input, dim=-1, descending=False, *, stable=False, out=None) -> (Tensor, LongTensor)

    Sorts the elements of the :attr:`input` tensor along a given dimension
    in ascending order by value.

    If :attr:`dim` is not given, the last dimension of the `input` is chosen.

    If :attr:`descending` is ``True`` then the elements are sorted in descending
    order by value.

    If :attr:`stable` is ``True`` then the sorting routine becomes stable, preserving
    the order of equivalent elements.

    A namedtuple of (values, indices) is returned, where the `values` are the
    sorted values and `indices` are the indices of the elements in the original
    `input` tensor.

    Args:
        input (Tensor): the input tensor.
        dim (int, optional): the dimension to sort along
        descending (bool, optional): controls the sorting order (ascending or descending)

    Keyword args:
        stable (bool, optional): makes the sorting routine stable, which guarantees that the order
            of equivalent elements is preserved.
        out (tuple, optional): the output tuple of (`Tensor`, `LongTensor`) that can
            be optionally given to be used as output buffers

    Example::

        >>> x = torch.randn(3, 4)
        >>> sorted, indices = torch.sort(x)
        >>> sorted
        tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
                [-0.5793,  0.0061,  0.6058,  0.9497],
                [-0.5071,  0.3343,  0.9553,  1.0960]])
        >>> indices
        tensor([[ 1,  0,  2,  3],
                [ 3,  1,  0,  2],
                [ 0,  3,  1,  2]])

        >>> sorted, indices = torch.sort(x, 0)
        >>> sorted
        tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
                [ 0.0608,  0.0061,  0.9497,  0.3343],
                [ 0.6058,  0.9553,  1.0960,  2.3332]])
        >>> indices
        tensor([[ 2,  0,  0,  1],
                [ 0,  1,  1,  2],
                [ 1,  2,  2,  0]])
        >>> x = torch.tensor([0, 1] * 9)
        >>> x.sort()
        torch.return_types.sort(
            values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            indices=tensor([ 2, 16,  4,  6, 14,  8,  0, 10, 12,  9, 17, 15, 13, 11,  7,  5,  3,  1]))
        >>> x.sort(stable=True)
        torch.return_types.sort(
            values=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            indices=tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16,  1,  3,  5,  7,  9, 11, 13, 15, 17]))
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1, descending: bool = False) -> torch.Tensor:
        """
        Sorts the input tensor along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int, optional): The dimension to sort along.
            descending (bool, optional): Controls the sorting order (ascending by default).

        Returns:
            torch.Tensor: Sorted tensor with same shape as input.
        """
        return torch.sort(x, dim=dim, descending=descending)[0]


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "8_Sort.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        descending_info = inputs[2]
        
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
        descending = descending_info["value"]
        input_groups.append([x, dim, descending])
    return input_groups


def get_init_inputs():
    return []
