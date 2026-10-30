import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    torch.histogram = histogram(input, bins, *, range=None, weight=None, density=False) -> (Tensor, Tensor)

    Computes a histogram of the values in a tensor.

    Supports N-dimensional input tensors, treating the tensor as a flattened array.

    Args:
        input (Tensor): the input tensor.
        bins (int or Tensor): If int, defines the number of equal-width bins in the given range.
            If Tensor, defines the sequence of bin edges including the rightmost edge.
        range (tuple of float, optional): Defines the range of the bins. Default: None.
        weight (Tensor, optional): If provided, weight each sample. Default: None.
        density (bool, optional): If False, the result will contain the count (or total weight)
            of samples in each bin. If True, the result is the probability density function
            at the bin, normalized such that the integral over the range is 1. Default: False.

    Returns:
        tuple: (hist, bin_edges) where hist is the histogram tensor and bin_edges is the
            tensor of bin edges.

    Examples:
        >>> torch.histogram(torch.tensor([1., 2, 1]), bins=4, range=(0., 3.))
        (tensor([0., 5., 2., 0.]), tensor([0., 0.75, 1.5, 2.25, 3.]))
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, bins, range=None, weight=None, density: bool = False) -> tuple:
        """
        Computes a histogram of the values in a tensor.

        Args:
            input (torch.Tensor): Input tensor.
            bins (int or torch.Tensor): Number of bins or bin edges.
            range (tuple, optional): Bin range (min, max). Default: None.
            weight (torch.Tensor, optional): Weight tensor. Default: None.
            density (bool, optional): Density normalization. Default: False.

        Returns:
            tuple: (hist, bin_edges).
        """
        return torch.histogram(input, bins, range=range, weight=weight, density=density)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "10_Histogram.json")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    fill_map = {
        'inf': float('inf'),
        '-inf': float('-inf'),
        'nan': float('nan'),
        '-nan': -float('nan'),
        'zero': 0.0,
    }

    input_groups = []

    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        scene = case.get("scene", "")

        input_info = inputs[0]
        bins_info = inputs[1]
        range_info = inputs[2] if len(inputs) > 2 else {"value": None}
        weight_info = inputs[3] if len(inputs) > 3 else {"value": None}
        density_info = inputs[4] if len(inputs) > 4 else {"value": False}

        input_dtype = dtype_map[input_info["dtype"]]
        input_shape = input_info["shape"]
        input_fill = input_info.get("fill", None)

        if input_fill in fill_map:
            input_tensor = torch.full(input_shape, fill_map[input_fill], dtype=input_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, input_shape, dtype=torch.float32)
            input_tensor = raw.to(input_dtype)
        else:
            raw = torch.empty(input_shape, dtype=torch.float32).uniform_(-5, 5)
            input_tensor = raw.to(input_dtype)

        # bins can be int or tensor
        if "dtype" in bins_info:
            bins_dtype = dtype_map[bins_info["dtype"]]
            bins_shape = bins_info["shape"]
            bins_fill = bins_info.get("fill", None)
            if bins_fill in fill_map:
                bins = torch.full(bins_shape, fill_map[bins_fill], dtype=bins_dtype)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, bins_shape, dtype=torch.float32)
                bins = raw.to(bins_dtype)
            else:
                raw = torch.empty(bins_shape, dtype=torch.float32).uniform_(-5, 5)
                bins = raw.to(bins_dtype)
        else:
            bins = bins_info["value"]

        range_val = range_info["value"]
        if range_val is not None:
            range_val = tuple(range_val)

        # weight optional
        weight = None
        if weight_info["value"] is not None:
            weight_dtype = dtype_map[weight_info["dtype"]]
            weight_shape = weight_info["shape"]
            weight_fill = weight_info.get("fill", None)
            if weight_fill in fill_map:
                weight = torch.full(weight_shape, fill_map[weight_fill], dtype=weight_dtype)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, weight_shape, dtype=torch.float32)
                weight = raw.to(weight_dtype)
            else:
                raw = torch.empty(weight_shape, dtype=torch.float32).uniform_(-5, 5)
                weight = raw.to(weight_dtype)

        density = density_info["value"]

        input_groups.append([input_tensor, bins, range_val, weight, density])

    return input_groups


def get_init_inputs():
    return []