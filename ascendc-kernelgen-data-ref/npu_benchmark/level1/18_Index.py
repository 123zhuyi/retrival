import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that selects elements from a tensor along a dimension using indices.
    torch.index_select = index_select(...)
    index_select(input, dim, index, *, out=None) -> Tensor

    Returns a new tensor which indexes the :attr:`input` tensor along dimension
    :attr:`dim` using the entries in :attr:`index`.

    The returned tensor has the same number of dimensions as the original tensor
    (:attr:`input`).  The :attr:`dim`\ th dimension has the same size as the length
    of :attr:`index`; other dimensions have the same size as in the original tensor.

    .. note:: The returned tensor does **not** use the same storage as the original
              tensor.  If :attr:`out` has a different shape than expected, we
              silently change it to the correct shape, reallocating the underlying
              storage if necessary.

    Args:
        input (Tensor): the input tensor.
        dim (int): the dimension in which we index
        index (IntTensor or LongTensor): the 1-D tensor containing the indices to index

    Keyword args:
        out (Tensor, optional): the output tensor.

    Example::

        >>> x = torch.randn(3, 4)
        >>> x
        tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-0.4664,  0.2647, -0.1228, -1.1068],
                [-1.1734, -0.6571,  0.7230, -0.6004]])
        >>> indices = torch.tensor([0, 2])
        >>> torch.index_select(x, 0, indices)
        tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-1.1734, -0.6571,  0.7230, -0.6004]])
        >>> torch.index_select(x, 1, indices)
        tensor([[ 0.1427, -0.5414],
                [-0.4664, -0.1228],
                [-1.1734,  0.7230]])
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
        """
        Selects elements from input tensor along the specified dimension using index.

        Args:
            x (torch.Tensor): Input tensor.
            dim (int): The dimension in which to index.
            index (torch.Tensor): The 1-D tensor containing the indices to index.

        Returns:
            torch.Tensor: Tensor with selected elements.
        """
        return torch.index_select(x, dim, index)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "18_Index.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        index_info = inputs[2]
        
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
        index_size = index_info["shape"][0]
        index = torch.randint(index_range[0], index_range[1] + 1, (index_size,), dtype=torch.int64)
        
        input_groups.append([x, dim, index])
    return input_groups


def get_init_inputs():
    return []
