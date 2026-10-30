import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that repeats elements of a tensor.
    torch.repeat_interleave = repeat_interleave(...)
    repeat_interleave(input, repeats, dim=None, *, output_size=None) -> Tensor

    Repeat elements of a tensor.

    .. warning::

        This is different from :meth:`torch.Tensor.repeat` but similar to ``numpy.repeat``.

    Args:
        input (Tensor): the input tensor.
        repeats (Tensor or int): The number of repetitions for each element.
            repeats is broadcasted to fit the shape of the given axis.
        dim (int, optional): The dimension along which to repeat values.
            By default, use the flattened input array, and return a flat output
            array.

    Keyword args:
        output_size (int, optional): Total output size for the given axis
            ( e.g. sum of repeats). If given, it will avoid stream synchronization
            needed to calculate output shape of the tensor.

    Returns:
        Tensor: Repeated tensor which has the same shape as input, except along the given axis.

    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> x.repeat_interleave(2)
        tensor([1, 1, 2, 2, 3, 3])
        >>> y = torch.tensor([[1, 2], [3, 4]])
        >>> torch.repeat_interleave(y, 2)
        tensor([1, 1, 2, 2, 3, 3, 4, 4])
        >>> torch.repeat_interleave(y, 3, dim=1)
        tensor([[1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4]])
        >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0)
        tensor([[1, 2],
                [3, 4],
                [3, 4]])
        >>> torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0, output_size=3)
        tensor([[1, 2],
                [3, 4],
                [3, 4]])

    If the `repeats` is `tensor([n1, n2, n3, ...])`, then the output will be
    `tensor([0, 0, ..., 1, 1, ..., 2, 2, ..., ...])` where `0` appears `n1` times,
    `1` appears `n2` times, `2` appears `n3` times, etc.

    .. function:: repeat_interleave(repeats, *) -> Tensor
       :noindex:

    Repeats 0 repeats[0] times, 1 repeats[1] times, 2 repeats[2] times, etc.

    Args:
        repeats (Tensor): The number of repetitions for each element.

    Returns:
        Tensor: Repeated tensor of size `sum(repeats)`.

    Example::

        >>> torch.repeat_interleave(torch.tensor([1, 2, 3]))
        tensor([0, 1, 1, 2, 2, 2])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats, dim: int = None, output_size: int = None) -> torch.Tensor:
        """
        Repeats elements of a tensor.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (int or torch.Tensor): Number of repetitions for each element.
            dim (int, optional): The dimension along which to repeat values.
            output_size (int, optional): Total output size for the repeated dimension.

        Returns:
            torch.Tensor: Tensor with repeated elements.
        """
        if output_size is not None:
            return torch.repeat_interleave(x, repeats, dim=dim, output_size=output_size)
        return torch.repeat_interleave(x, repeats, dim=dim)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "23_RepeatInterleave.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        repeats_info = inputs[1]
        dim_info = inputs[2]
        
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
        repeats = repeats_info["value"]
        dim = dim_info["value"]
        input_groups.append([x, repeats, dim])
    return input_groups


def get_init_inputs():
    return []
