import torch
import torch.nn as nn
import torch_npu
import json
import os


class Model(nn.Module):
    r"""
        Simple model that applies Group Normalization.
        torch.nn.functional.group_norm = group_norm(
        input: Tensor,
        num_groups: int,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        eps: float = 1e-05
    ) -> Tensor
        Apply Group Normalization for last certain number of dimensions.
        .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        The input channels are separated into num_groups groups, each containing
        num_channels / num_groups channels. num_channels must be divisible by
        num_groups. The mean and standard-deviation are calculated
        separately over each group. γ and β are learnable
        per-channel affine transform parameter vectors of size num_channels if
        affine is True.
        The variance is calculated via the biased estimator, equivalent to
        torch.var(input, correction=0).

        This layer uses statistics computed from input data in both training and
        evaluation modes.

        Args:
            num_groups (int): number of groups to separate the channels into
            num_channels (int): number of channels expected in input
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            affine: a boolean value that when set to True, this module
                has learnable per-channel affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: True
            bias: If set to False, the layer will not learn an additive bias (only relevant if
                affine is True). Default: True

        Shape:
            - Input: (N, C, *) where C=num_channels
            - Output: (N, C, *) (same shape as input)

        Examples::

            >>> input = torch.randn(20, 6, 10, 10)
            >>> # Separate 6 channels into 3 groups
            >>> m = nn.GroupNorm(3, 6)
            >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
            >>> m = nn.GroupNorm(6, 6)
            >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
            >>> m = nn.GroupNorm(1, 6)
            >>> # Activating the module
            >>> output = m(input)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, num_groups: int, weight: torch.Tensor = None,
                bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Group Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [N, C, *].
            num_groups (int): Number of groups to separate channels into.
            weight (torch.Tensor, optional): Weight tensor of shape [C].
            bias (torch.Tensor, optional): Bias tensor of shape [C].

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.group_norm(x, num_groups, weight=weight, bias=bias)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "11_GroupNorm.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-50, 50).item())
            sigma = float(torch.empty(1).uniform_(4, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5)

        num_groups = None
        weight = None
        bias = None

        for inp in inputs[1:]:
            if inp["name"] == "num_groups":
                num_groups = inp["value"]
            elif inp["name"] == "weight":
                weight = torch.randn(inp["shape"], dtype=dtype)
            elif inp["name"] == "bias":
                bias = torch.randn(inp["shape"], dtype=dtype)

        input_groups.append([x, num_groups, weight, bias])
    return input_groups


def get_init_inputs():
    return []
