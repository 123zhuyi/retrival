import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    Simple model that performs element-wise addition with broadcasting support.
    torch.add = add(...)
    add(input, other, *, alpha=1, out=None) -> Tensor

    Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

    .. math::
        \text{{out}}_i = \text{{input}}_i + \text{{alpha}} \times \text{{other}}_i


    Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
    :ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

    Args:
        input (Tensor): the input tensor.
        other (Tensor or Number): the tensor or number to add to :attr:`input`.

    Keyword arguments:
        alpha (Number): the multiplier for :attr:`other`.
        out (Tensor, optional): the output tensor.

    Examples::

        >>> a = torch.randn(4)
        >>> a
        tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
        >>> torch.add(a, 20)
        tensor([ 20.0202,  21.0985,  21.3506,  19.3944])

        >>> b = torch.randn(4)
        >>> b
        tensor([-0.9732, -0.3497,  0.6245,  0.4022])
        >>> c = torch.randn(4, 1)
        >>> c
        tensor([[ 0.3743],
                [-1.7724],
                [-0.5811],
                [-0.8017]])
        >>> torch.add(b, c, alpha=10)
        tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
                [-18.6971, -18.0736, -17.0994, -17.3216],
                [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
                [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Applies element-wise addition to the input tensors with broadcasting support.

        Args:
            x (torch.Tensor): First input tensor of any shape.
            y (torch.Tensor): Second input tensor, broadcastable with x.
            alpha (float, optional): The multiplier for y.

        Returns:
            torch.Tensor: Output tensor x + alpha * y, shape follows broadcasting rules.
        """
        return torch.add(x, y, alpha=alpha)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_Add.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        y_info = inputs[1]
        alpha_info = inputs[2]
        
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
        y = torch.randn(y_info["shape"], dtype=dtype)
        alpha = alpha_info["value"]
        input_groups.append([x, y, alpha])
    return input_groups


def get_init_inputs():
    return []
