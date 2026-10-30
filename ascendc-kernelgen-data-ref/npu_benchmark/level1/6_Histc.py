import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that computes the histogram of a tensor.
    torch.histc = histc(...)
    histc(input, bins=100, min=0, max=0, *, out=None) -> Tensor

    Computes the histogram of a tensor.

    The elements are sorted into equal width bins between :attr:`min` and
    :attr:`max`. If :attr:`min` and :attr:`max` are both zero, the minimum and
    maximum values of the data are used.

    Elements lower than min and higher than max and ``NaN`` elements are ignored.

    Args:
        input (Tensor): the input tensor.
        bins (int): number of histogram bins
        min (Scalar): lower end of the range (inclusive)
        max (Scalar): upper end of the range (inclusive)

    Keyword args:
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: Histogram represented as a tensor

    Example::

        >>> torch.histc(torch.tensor([1., 2, 1]), bins=4, min=0, max=3)
        tensor([ 0.,  2.,  1.,  0.])
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, bins: int, min: int, max: int) -> torch.Tensor:
        """
        Computes the histogram of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.
            bins (int): Number of histogram bins.
            min (int): Lower end of the range (inclusive).
            max (int): Upper end of the range (inclusive).

        Returns:
            torch.Tensor: Histogram tensor of shape (bins,).
        """
        return torch.histc(x, bins=bins, min=min, max=max)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_Histc.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        bins_info = inputs[1]
        min_info = inputs[2]
        max_info = inputs[3]
        
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
        bins = bins_info["value"]
        min_val = min_info["value"]
        max_val = max_info["value"]
        input_groups.append([x, bins, min_val, max_val])
    return input_groups


def get_init_inputs():
    return []
