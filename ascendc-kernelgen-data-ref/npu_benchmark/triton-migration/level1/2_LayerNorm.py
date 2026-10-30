import torch
import torch.nn as nn
import torch_npu
import json
import os

class Model(nn.Module):
    r"""
        Simple model that applies Layer Normalization.
        torch.nn.functional.layer_norm = layer_norm(
        input: Tensor,
        normalized_shape: list[int],
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        eps: float = 1e-05
    ) -> Tensor
        Apply Layer Normalization for last certain number of dimensions.
        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
            The mean and standard-deviation are calculated over the last D dimensions, where D
        is the dimension of normalized_shape. For example, if normalized_shape
        is (3, 5) (a 2-dimensional shape), the mean and standard-deviation are computed over
        the last 2 dimensions of the input (i.e. input.mean((-2, -1))).
        γ and β are learnable affine transform parameters of
        normalized_shape if elementwise_affine is True.
        The variance is calculated via the biased estimator, equivalent to
        torch.var(input, correction=0).

        .. note::
            Unlike Batch Normalization and Instance Normalization, which applies
            scalar scale and bias for each entire channel/plane with the
            affine option, Layer Normalization applies per-element scale and
            bias with elementwise_affine.

        This layer uses statistics computed from input data in both training and
        evaluation modes.

        Args:
            normalized_shape (int or list or torch.Size): input shape from an expected input
                of size

                .. math::
                    [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                        \times \ldots \times \text{normalized\_shape}[-1]]

                If a single integer is used, it is treated as a singleton list, and this module will
                normalize over the last dimension which is expected to be of that specific size.
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: a boolean value that when set to True, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: True
            bias: If set to False, the layer will not learn an additive bias (only relevant if
                elementwise_affine is True). Default: True

        Attributes:
            weight: the learnable weights of the module of shape
                normalized_shape when elementwise_affine is set to True.
                The values are initialized to 1.
            bias:   the learnable bias of the module of shape
                    normalized_shape when elementwise_affine is set to True.
                    The values are initialized to 0.

        Shape:
            - Input: (N, *)
            - Output: (N, *) (same shape as input)

        Examples::

            >>> # NLP Example
            >>> batch, sentence_length, embedding_dim = 20, 5, 10
            >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
            >>> layer_norm = nn.LayerNorm(embedding_dim)
            >>> # Activate module
            >>> layer_norm(embedding)
            >>>
            >>> # Image Example
            >>> N, C, H, W = 20, 5, 10, 10
            >>> input = torch.randn(N, C, H, W)
            >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
            >>> # as shown in the image below
            >>> layer_norm = nn.LayerNorm([C, H, W])
            >>> output = layer_norm(input)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, normalized_shape: list, weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """
        Applies Layer Normalization over a mini-batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape [*, normalized_shape[0], ...].
            normalized_shape (list): Shape over which to normalize.
            weight (torch.Tensor, optional): Weight tensor of shape normalized_shape.
            bias (torch.Tensor, optional): Bias tensor of shape normalized_shape.

        Returns:
            torch.Tensor: Normalized tensor with same shape as input.
        """
        return torch.nn.functional.layer_norm(x, normalized_shape, weight=weight, bias=bias)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_LayerNorm.json")
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
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5)
        
        normalized_shape = None
        weight = None
        bias = None
        
        for inp in inputs[1:]:
            if inp["name"] == "normalized_shape":
                normalized_shape = inp["value"]
            elif inp["name"] == "weight":
                weight = torch.randn(inp["shape"], dtype=dtype)
            elif inp["name"] == "bias":
                bias = torch.randn(inp["shape"], dtype=dtype)
        
        input_groups.append([x, normalized_shape, weight, bias])
    return input_groups


def get_init_inputs():
    return []
