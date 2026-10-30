import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    Simple model that performs batch matrix multiplication.
    torch.bmm = bmm(...)
    bmm(input, mat2, out_dtype=None, *, out=None) -> Tensor

    Performs a batch matrix-matrix product of matrices stored in :attr:`input`
    and :attr:`mat2`.

    :attr:`input` and :attr:`mat2` must be 3-D tensors each containing
    the same number of matrices.

    If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
    :math:`(b \times m \times p)` tensor, :attr:`out` will be a
    :math:`(b \times n \times p)` tensor.

    .. math::
        \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
              For broadcasting matrix products, see :func:`torch.matmul`.

    Args:
        input (Tensor): the first batch of matrices to be multiplied
        mat2 (Tensor): the second batch of matrices to be multiplied
        out_dtype (dtype, optional): the dtype of the output tensor,

    Keyword Args:
        out (Tensor, optional): the output tensor.

    Example::

        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> input = torch.normal(mu, sigma, >>> input_info["shape"], dtype=dtype)
        else:
            >>> input = torch.empty(>>> input_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> mat2 = torch.normal(mu, sigma, >>> mat2_info["shape"], dtype=dtype)
        else:
            >>> mat2 = torch.empty(>>> mat2_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        >>> res = torch.bmm(input, mat2)
        >>> res.size()
        torch.Size([10, 3, 5])
    """

    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Applies batch matrix multiplication between A and B.

        Args:
            A (torch.Tensor): Input tensor of shape (batch, m, n).
            B (torch.Tensor): Input tensor of shape (batch, n, p).

        Returns:
            torch.Tensor: Output tensor of shape (batch, m, p) after performing torch.bmm.
        """
        return torch.bmm(A, B)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "1_BatchMatmul.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        A_info = inputs[0]
        B_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[A_info["dtype"]]
        
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            A = torch.normal(mu, sigma, A_info["shape"], dtype=dtype)
        else:
            A = torch.empty(A_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            B = torch.normal(mu, sigma, B_info["shape"], dtype=dtype)
        else:
            B = torch.empty(B_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        input_groups.append([A, B])
    return input_groups


def get_init_inputs():
    return []
