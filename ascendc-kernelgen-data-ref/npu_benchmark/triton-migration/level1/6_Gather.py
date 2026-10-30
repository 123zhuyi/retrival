import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that gathers values along a dimension using indices.
    torch.gather = gather(...)
    gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor

    Gathers values along an axis specified by `dim`.

    For a 3-D tensor the output is specified by::

        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :attr:`input` and :attr:`index` must have the same number of dimensions.
    It is also required that ``index.size(d) <= input.size(d)`` for all
    dimensions ``d != dim``.  :attr:`out` will have the same shape as :attr:`index`.
    Note that ``input`` and ``index`` do not broadcast against each other.
    When :attr:`index` is empty, we always return an empty output with the same shape
    without further error checking.

    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to gather

    Keyword arguments:
        sparse_grad (bool, optional): If ``True``, gradient w.r.t. :attr:`input` will be a sparse tensor.
        out (Tensor, optional): the destination tensor

    Example::

        >>> t = torch.tensor([[1, 2], [3, 4]])
        >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
        tensor([[ 1,  1],
                [ 4,  3]])
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor, sparse_grad: bool = False) -> torch.Tensor:
        """
        Gathers values along an axis specified by dim.

        Args:
            x (torch.Tensor): Input tensor (src).
            dim (int): The axis along which to index.
            index (torch.Tensor): The indices of elements to gather.
            sparse_grad (bool, optional): If True, gradient w.r.t. input will be a sparse tensor.

        Returns:
            torch.Tensor: Tensor with gathered values, same shape as index.
        """
        return torch.gather(x, dim, index, sparse_grad=sparse_grad)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_Gather.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        index_info = inputs[2]
        sparse_grad_info = inputs[3]
        
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
        
        index_range = index_info.get("range", [0, x_info["shape"][dim] - 1])
        index = torch.randint(index_range[0], index_range[1] + 1, tuple(index_info["shape"]), dtype=torch.int64)
        sparse_grad = sparse_grad_info["value"]
        
        input_groups.append([x, dim, index, sparse_grad])
    return input_groups


def get_init_inputs():
    return []
