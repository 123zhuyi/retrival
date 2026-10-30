import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    Simple model that performs 3D average pooling.
    torch.nn.functional.avg_pool3d = avg_pool3d(...)
    avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor

    Applies 3D average-pooling operation in :math:`kT \times kH \times kW` regions by step
    size :math:`sT \times sH \times sW` steps. The number of output features is equal to
    :math:`\lfloor\frac{\text{input planes}}{sT}\rfloor`.

    See :class:`~torch.nn.AvgPool3d` for details and output shape.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in\_channels} , iT \times iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kT, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padT, padH, padW)`. Should be at most half
          of effective kernel size, that is :math:`((kernelSize - 1) * dilation + 1) / 2`.
          Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation
        divisor_override: if specified, it will be used as divisor, otherwise
            size of the pooling region will be used. Default: None
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, kernel_size, stride=None, padding: int = 0,
                ceil_mode: bool = False, count_include_pad: bool = True,
                divisor_override=None) -> torch.Tensor:
        """
        Applies 3D average pooling over an input signal.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            kernel_size: Size of the pooling window.
            stride (optional): Stride of the pooling window. Default: kernel_size.
            padding (int, optional): Implicit zero padding.
            ceil_mode (bool, optional): Use ceil instead of floor for output shape.
            count_include_pad (bool, optional): Include zero-padding in averaging.
            divisor_override (optional): If specified, will be used as divisor.

        Returns:
            torch.Tensor: Pooled tensor.
        """
        return torch.nn.functional.avg_pool3d(
            x, kernel_size, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "26_AvgPool3d.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        kernel_size_info = inputs[1]
        stride_info = inputs[2]
        padding_info = inputs[3]
        
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
        kernel_size = kernel_size_info["value"]
        stride = stride_info["value"]
        padding = padding_info["value"]
        input_groups.append([x, kernel_size, stride, padding])
    return input_groups


def get_init_inputs():
    return []
