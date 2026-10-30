import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that repeats a tensor along specified dimensions.
    torch.Tensor.repeat = repeat(...) unbound torch._C.TensorBase method
    repeat(*repeats) -> Tensor

    Repeats this tensor along the specified dimensions.

    Unlike :meth:`~Tensor.expand`, this function copies the tensor's data.

    .. warning::

        :meth:`~Tensor.repeat` behaves differently from
        `numpy.repeat
        but is more similar to
        `numpy.tile
        For the operator similar to `numpy.repeat`, see :func:`torch.repeat_interleave`.

    Args:
        repeat (torch.Size, int..., tuple of int or list of int): The number of times to repeat this tensor along each dimension

    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> x.repeat(4, 2)
        tensor([[ 1,  2,  3,  1,  2,  3],
                [ 1,  2,  3,  1,  2,  3],
                [ 1,  2,  3,  1,  2,  3],
                [ 1,  2,  3,  1,  2,  3]])
        >>> x.repeat(4, 2, 1).size()
        torch.Size([4, 2, 3])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, repeats: tuple) -> torch.Tensor:
        """
        Repeats the tensor along each dimension.

        Args:
            x (torch.Tensor): Input tensor.
            repeats (tuple): Number of repeats for each dimension.

        Returns:
            torch.Tensor: Repeated tensor.
        """
        return x.repeat(*repeats)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "16_Repeat.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        repeats_info = inputs[1]
        
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
        repeats = tuple(repeats_info["value"])
        input_groups.append([x, repeats])
    return input_groups


def get_init_inputs():
    return []
