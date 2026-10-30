import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that performs a permutation of tensor dimensions.
    torch.permute = permute(...)
    permute(input, dims) -> Tensor

    Returns a view of the original tensor :attr:`input` with its dimensions permuted.

    Args:
        input (Tensor): the input tensor.
        dims (torch.Size, tuple of int or list of int): the desired ordering of dimensions.

    Example:
        >>> x = torch.randn(2, 3, 5)
        >>> x.size()
        torch.Size([2, 3, 5])
        >>> torch.permute(x, (2, 0, 1)).size()
        torch.Size([5, 2, 3])

    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dims: tuple) -> torch.Tensor:
        """
        Permutes the dimensions of the input tensor.

        Args:
            x (torch.Tensor): Input tensor with at least the number of dimensions in dims.
            dims (tuple): The desired ordering of dimensions.

        Returns:
            torch.Tensor: Tensor with permuted dimensions.
        """
        return torch.permute(x, dims).contiguous()


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "12_Permute.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        dims_info = inputs[1]
        
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
        dims = tuple(dims_info["value"])
        input_groups.append([x, dims])
    return input_groups


def get_init_inputs():
    return []
