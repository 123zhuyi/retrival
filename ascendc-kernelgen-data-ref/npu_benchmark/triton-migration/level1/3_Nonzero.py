import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    Simple model that returns indices of non-zero elements.
    torch.nonzero = nonzero(...)
    nonzero(input, *, out=None, as_tuple=False) -> LongTensor or tuple of LongTensors

    .. note::
        :func:`torch.nonzero(..., as_tuple=False) <torch.nonzero>` (default) returns a
        2-D tensor where each row is the index for a nonzero value.

        :func:`torch.nonzero(..., as_tuple=True) <torch.nonzero>` returns a tuple of 1-D
        index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
        gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
        contains nonzero indices for a certain dimension.

        See below for more details on the two behaviors.

    **When** :attr:`as_tuple` **is** ``False`` **(default)**:

    Returns a tensor containing the indices of all non-zero elements of
    :attr:`input`.  Each row in the result contains the indices of a non-zero
    element in :attr:`input`. The result is sorted lexicographically, with
    the last index changing the fastest (C-style).

    If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
    :attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
    non-zero elements in the :attr:`input` tensor.

    **When** :attr:`as_tuple` **is** ``True``:

    Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
    each containing the indices (in that dimension) of all non-zero elements of
    :attr:`input` .

    If :attr:`input` has :math:`n` dimensions, then the resulting tuple contains :math:`n`
    tensors of size :math:`z`, where :math:`z` is the total number of
    non-zero elements in the :attr:`input` tensor.

    As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
    value, it is treated as a one-dimensional tensor with one element.

    Args:
        input (Tensor): the input tensor.

    Keyword args:
        out (LongTensor, optional): the output tensor containing indices

    Returns:
        LongTensor or tuple of LongTensor: If :attr:`as_tuple` is ``False``, the output
        tensor containing indices. If :attr:`as_tuple` is ``True``, one 1-D tensor for
        each dimension, containing the indices of each nonzero element along that
        dimension.

    Example::

        >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]))
        tensor([[ 0],
                [ 1],
                [ 2],
                [ 4]])
        >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
        ...                             [0.0, 0.4, 0.0, 0.0],
        ...                             [0.0, 0.0, 1.2, 0.0],
        ...                             [0.0, 0.0, 0.0,-0.4]]))
        tensor([[ 0,  0],
                [ 1,  1],
                [ 2,  2],
                [ 3,  3]])
        >>> torch.nonzero(torch.tensor([1, 1, 1, 0, 1]), as_tuple=True)
        (tensor([0, 1, 2, 4]),)
        >>> torch.nonzero(torch.tensor([[0.6, 0.0, 0.0, 0.0],
        ...                             [0.0, 0.4, 0.0, 0.0],
        ...                             [0.0, 0.0, 1.2, 0.0],
        ...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
        (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
        >>> torch.nonzero(torch.tensor(5), as_tuple=True)
        (tensor([0]),)
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, as_tuple: bool = False):
        """
        Returns indices of non-zero elements in the tensor.

        Args:
            x (torch.Tensor): Input tensor.
            as_tuple (bool, optional): If True, returns a tuple of 1-D tensors, one for each dimension.

        Returns:
            torch.Tensor or tuple: Indices of non-zero elements.
        """
        return torch.nonzero(x, as_tuple=as_tuple)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_Nonzero.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        as_tuple_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        dtype = dtype_map.get(x_info["dtype"], torch.float32)
        
        if dtype in [torch.int32, torch.int64]:
            x = torch.randint(-10, 10, x_info["shape"], dtype=dtype)
        else:
            if idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype)
            else:
                x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5)
        as_tuple = as_tuple_info["value"]
        input_groups.append([x, as_tuple])
    return input_groups


def get_init_inputs():
    return []
