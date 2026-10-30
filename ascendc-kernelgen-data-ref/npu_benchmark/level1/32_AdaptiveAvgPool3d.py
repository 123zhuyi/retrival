import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    torch.nn.AdaptiveAvgPool3d = class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd)
    torch.nn.AdaptiveAvgPool3d(output_size: _size_any_opt_t) -> None

    Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form D x H x W.
                        Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                        D, H and W can be either a ``int``, or ``None`` which means the size will
                        be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})` or :math:`(C, S_{0}, S_{1}, S_{2})`,
            where :math:`S=\text{output\_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveAvgPool3d((5, 7, 9))
        >>> input = torch.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = torch.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, output_size) -> torch.Tensor:
        """
        Applies 3D adaptive average pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            output_size: Target output size of the form D x H x W.
                         Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                         D, H and W can be either an int, or None which means the size will
                         be the same as that of the input.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        m = nn.AdaptiveAvgPool3d(output_size)
        return m(x)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "32_AdaptiveAvgPool3d.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for idx, case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        output_size_info = inputs[1]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-50, 50).item())
            sigma = float(torch.empty(1).uniform_(4, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype) + torch.ones(x_info["shape"], dtype=dtype)
            x = torch.clamp(x, mu - 4*sigma + 1, mu + 4*sigma + 1)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5) + torch.ones(x_info["shape"], dtype=dtype)
        output_size = output_size_info["value"]
        if isinstance(output_size, list):
            output_size = tuple(output_size)
        input_groups.append([x, output_size])
    return input_groups


def get_init_inputs():
    return []