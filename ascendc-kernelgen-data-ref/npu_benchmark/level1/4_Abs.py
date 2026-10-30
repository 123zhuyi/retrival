import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
     Simple model that performs absolute value operation.
    torch.abs = abs(...)
    abs(input: Tensor, *, out: Optional[Tensor]) -> Tensor

    Computes the absolute value of each element in :attr:`input`.

    .. math::
        \text{out}_{i} = |\text{input}_{i}|

    Args:
        input (Tensor): the input tensor.

    Keyword args:
        out (Tensor, optional): the output tensor.

    Example::

        >>> torch.abs(torch.tensor([-1, -2, 3]))
        tensor([ 1,  2,  3])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies absolute value to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with absolute values, same shape as input.
        """
        return torch.abs(x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_Abs.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        
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
        input_groups.append([x])
    return input_groups


def get_init_inputs():
    return []
