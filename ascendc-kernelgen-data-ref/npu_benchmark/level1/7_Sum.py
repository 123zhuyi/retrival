import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the sum of elements along specified dimensions.
    torch.sum = sum(...)
    sum(input, *, dtype=None) -> Tensor

    Returns the sum of all elements in the :attr:`input` tensor.

    Args:
        input (Tensor): the input tensor.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.

    .. note:: Use the `dtype` argument if you need the result in a specific tensor type.
              Otherwise, the result type may be automatically promoted (e.g., from `torch.int32` to `torch.int64`).

    Example::

        >>> a = torch.randn(1, 3)
        >>> a
        tensor([[ 0.1133, -0.9567,  0.2958]])
        >>> torch.sum(a)
        tensor(-0.5475)

    .. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
       :noindex:

    Returns the sum of each row of the :attr:`input` tensor in the given
    dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
    reduce over all of them.


    If :attr:`keepdim` is ``True``, the output tensor is of the same size
    as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
    Otherwise, :attr:`dim` is squeezed (see :func:`torch.squeeze`), resulting in the
    output tensor having 1 (or ``len(dim)``) fewer dimension(s).


    Args:
        input (Tensor): the input tensor.

        dim (int or tuple of ints, optional): the dimension or dimensions to reduce.
            If ``None``, all dimensions are reduced.


        keepdim (bool, optional): whether the output tensor has :attr:`dim` retained or not. Default: ``False``.


    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.

    Example::

        >>> a = torch.randn(4, 4)
        >>> a
        tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
                [-0.2993,  0.9138,  0.9337, -1.6864],
                [ 0.1132,  0.7892, -0.1003,  0.5688],
                [ 0.3637, -0.9906, -0.4752, -1.5197]])
        >>> torch.sum(a, 1)
        tensor([-0.4598, -0.1381,  1.3708, -2.6217])
        >>> b = torch.arange(4 * 5 * 6).view(4, 5, 6)
        >>> torch.sum(b, (2, 1))
        tensor([  435.,  1335.,  2235.,  3135.])
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
        """
        Returns the sum of elements along specified dimensions.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            dim (int or tuple of ints, optional): Dimension(s) to reduce.
            keepdim (bool): Whether to keep the reduced dimension(s).

        Returns:
            torch.Tensor: Tensor with sum along specified dimensions.
        """
        return torch.sum(x, dim=dim, keepdim=keepdim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_Sum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        keepdim_info = inputs[2]
        
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
        keepdim = keepdim_info["value"]
        input_groups.append([x, dim, keepdim])
    return input_groups


def get_init_inputs():
    return []
