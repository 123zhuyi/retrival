import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    torch.nn.functional.gelu = gelu(...)
    gelu(input, approximate = 'none') -> Tensor

    When the approximate argument is 'none', it applies element-wise the function
    :math:`\text{GELU}(x) = x * \Phi(x)`
    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
    When the approximate argument is 'tanh', Gelu is estimated with

    .. math::
        \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))
    
    """
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, approximate='none') -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            approximate (str, optional): The gelu approximation algorithm to use: 'none'|'tanh'.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        return torch.nn.functional.gelu(x, approximate=approximate)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_GELU.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        approximate_info = inputs[1]
        
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
        approximate = approximate_info["value"]
        input_groups.append([x, approximate])
    return input_groups


def get_init_inputs():
    return []
