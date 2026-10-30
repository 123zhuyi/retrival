import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    Simple model that performs a SwiGLU activation.
    torch.nn.functional.silu = silu(input: Tensor, inplace: bool = False) -> Tensor
    Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Applies SwiGLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor where the size of dim must be even.
            dim (int, optional): The dimension along which to chunk the tensor.

        Returns:
            torch.Tensor: Output tensor with SwiGLU applied, shape is same as x except
                          dim is halved.
        """
        a, b = torch.chunk(x, 2, dim=dim)
        return torch.nn.functional.silu(a) * b


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_SwiGLU.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        dim_info = inputs[1]
        
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
        input_groups.append([x, dim])
    return input_groups


def get_init_inputs():
    return []
