import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that concatenates tensors along a dimension.
    torch.cat = cat(...)
    cat(tensors, dim=0, *, out=None) -> Tensor

    Concatenates the given sequence of tensors in :attr:`tensors` in the given dimension.
    All tensors must either have the same shape (except in the concatenating
    dimension) or be a 1-D empty tensor with size ``(0,)``.

    :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
    and :func:`torch.chunk`.

    :func:`torch.cat` can be best understood via examples.

    .. seealso::

        :func:`torch.stack` concatenates the given sequence along a new dimension.

    Args:
        tensors (sequence of Tensors): Non-empty tensors provided must have the same shape,
            except in the cat dimension.

        dim (int, optional): the dimension over which the tensors are concatenated

    Keyword args:
        out (Tensor, optional): the output tensor.

    Example::

        >>> x = torch.randn(2, 3)
        >>> x
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 0)
        tensor([[ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497],
                [ 0.6580, -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497]])
        >>> torch.cat((x, x, x), 1)
        tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
                 -1.0969, -0.4614],
                [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
                 -0.5790,  0.1497]])
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors: list, dim: int = 0) -> torch.Tensor:
        """
        Concatenates the given sequence of tensors in the given dimension.

        Args:
            tensors (list): List of tensors to concatenate. All tensors must have the same shape except in the concatenating dimension.
            dim (int, optional): The dimension over which the tensors are concatenated.

        Returns:
            torch.Tensor: Concatenated tensor.
        """
        return torch.cat(tensors, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "13_Cat.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        tensors_info = inputs[0]
        dim_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[tensors_info["dtype"]]
        
        shapes = tensors_info["shapes"]
        tensors = []
        for shape in shapes:
            if idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                tensors.append(torch.normal(mu, sigma, shape, dtype=dtype))
            else:
                tensors.append(torch.empty(shape, dtype=dtype).uniform_(-5, 5))
        dim = dim_info["value"]
        input_groups.append([tensors, dim])
    return input_groups


def get_init_inputs():
    return []
