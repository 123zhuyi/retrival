import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    torch.nn.functional.grid_sample = grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)

    Given an input and a flow-field grid, computes the output using input values
    and pixel locations from grid.

    Currently, only spatial (4-D) and volumetric (5-D) input are supported.

    For 4-D input with shape (N, C, H_in, W_in) and grid with shape
    (N, H_out, W_out, 2), the output will have shape (N, C, H_out, W_out).

    For 5-D input with shape (N, C, D_in, H_in, W_in) and grid with shape
    (N, D_out, H_out, W_out, 3), the output will have shape
    (N, C, D_out, H_out, W_out).

    grid specifies the sampling pixel locations normalized by the input spatial
    dimensions. It should have most values in the range of [-1, 1].

    Args:
        input (Tensor): 4-D (N,C,H,W) or 5-D (N,C,D,H,W) input tensor.
        grid (Tensor): 4-D (N,H_out,W_out,2) or 5-D (N,D_out,H_out,W_out,3) grid tensor.
        mode (str): 'bilinear' | 'nearest' | 'bicubic'. Default: 'bilinear'.
            bicubic only supports 4-D input.
        padding_mode (str): 'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool): Default: False.

    Returns:
        Tensor: output tensor.

    Example::

        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> grid = torch.FloatTensor([[[-1, -1], [1, 1]]]).unsqueeze(0)
        >>> torch.nn.functional.grid_sample(input, grid)
        tensor([[[[ 0., 15.]]]])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, grid, mode='bilinear', padding_mode='zeros',
                align_corners=False):
        return torch.nn.functional.grid_sample(
            input, grid, mode=mode, padding_mode=padding_mode,
            align_corners=align_corners
        )


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "4_GridSample.json")

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
        grid_info = inputs[1]

        input_dtype = dtype_map[input_info["dtype"]]
        grid_dtype = dtype_map[grid_info["dtype"]]
        input_shape = input_info["shape"]
        grid_shape = grid_info["shape"]
        input_fill = input_info.get("fill", None)
        grid_fill = grid_info.get("fill", None)

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

        if grid_fill in fill_map:
            grid_tensor = torch.full(grid_shape, fill_map[grid_fill], dtype=grid_dtype)
        elif idx % 2 == 0:
            grid_tensor = torch.empty(grid_shape, dtype=torch.float32).uniform_(-1, 1).to(grid_dtype)
        else:
            grid_tensor = torch.empty(grid_shape, dtype=torch.float32).uniform_(-1, 1).to(grid_dtype)

        mode = 'bilinear'
        padding_mode = 'zeros'
        align_corners = False

        for inp in inputs[2:]:
            if inp["name"] == "mode":
                mode = inp["value"]
            elif inp["name"] == "padding_mode":
                padding_mode = inp["value"]
            elif inp["name"] == "align_corners":
                align_corners = inp["value"]

        input_groups.append([input_tensor, grid_tensor, mode, padding_mode, align_corners])

    return input_groups


def get_init_inputs():
    return []