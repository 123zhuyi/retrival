import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
    Simple model that performs matrix multiplication with transposed B.
        torch.matmul = matmul(...)
    matmul(input, other, *, out=None) -> Tensor

    Matrix product of two tensors.

    The behavior depends on the dimensionality of the tensors as follows:

    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned.
    - If both arguments are at least 1-dimensional and at least one argument is
      N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
      argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
      batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
      1 is appended to its dimension for the purpose of the batched matrix multiply and removed after.

      The first N-2 dimensions of each argument, the batch dimensions, are
      :ref:`broadcast <broadcasting-semantics>` (and thus must be broadcastable).
      The last 2, the matrix dimensions, are handled as in the matrix-matrix product.

      For example, if :attr:`input` is a
      :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
      tensor, the batch dimensions are :math:`(j \times 1)` and :math:`(k)`,
      and the matrix dimensions are :math:`(n \times m)` and :math:`(m \times p)`.
      :attr:`out` will be a :math:`(j \times k \times n \times p)` tensor.

    This operation has support for arguments with :ref:`sparse layouts<sparse-docs>`. In particular the
    matrix-matrix (both arguments 2-dimensional) supports sparse arguments with the same restrictions
    as :func:`torch.mm`


    .. warning::
        Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,
        or may not have autograd support. If you notice missing functionality please
        open a feature request.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    .. note::

        The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

    Arguments:
        input (Tensor): the first tensor to be multiplied
        other (Tensor): the second tensor to be multiplied

    Keyword args:
        out (Tensor, optional): the output tensor.

    Example::

        >>> # vector x vector
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor1 = torch.normal(mu, sigma, >>> tensor1_info["shape"], dtype=dtype)
        else:
            >>> tensor1 = torch.empty(>>> tensor1_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor2 = torch.normal(mu, sigma, >>> tensor2_info["shape"], dtype=dtype)
        else:
            >>> tensor2 = torch.empty(>>> tensor2_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([])
        >>> # matrix x vector
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor1 = torch.normal(mu, sigma, >>> tensor1_info["shape"], dtype=dtype)
        else:
            >>> tensor1 = torch.empty(>>> tensor1_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor2 = torch.normal(mu, sigma, >>> tensor2_info["shape"], dtype=dtype)
        else:
            >>> tensor2 = torch.empty(>>> tensor2_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([3])
        >>> # batched matrix x broadcasted vector
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor1 = torch.normal(mu, sigma, >>> tensor1_info["shape"], dtype=dtype)
        else:
            >>> tensor1 = torch.empty(>>> tensor1_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor2 = torch.normal(mu, sigma, >>> tensor2_info["shape"], dtype=dtype)
        else:
            >>> tensor2 = torch.empty(>>> tensor2_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([10, 3])
        >>> # batched matrix x batched matrix
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor1 = torch.normal(mu, sigma, >>> tensor1_info["shape"], dtype=dtype)
        else:
            >>> tensor1 = torch.empty(>>> tensor1_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor2 = torch.normal(mu, sigma, >>> tensor2_info["shape"], dtype=dtype)
        else:
            >>> tensor2 = torch.empty(>>> tensor2_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([10, 3, 5])
        >>> # batched matrix x broadcasted matrix
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor1 = torch.normal(mu, sigma, >>> tensor1_info["shape"], dtype=dtype)
        else:
            >>> tensor1 = torch.empty(>>> tensor1_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            >>> tensor2 = torch.normal(mu, sigma, >>> tensor2_info["shape"], dtype=dtype)
        else:
            >>> tensor2 = torch.empty(>>> tensor2_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)
        >>> torch.matmul(tensor1, tensor2).size()
        torch.Size([10, 3, 5])
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Applies matrix multiplication between A and transposed B.

        Args:
            A (torch.Tensor): Input tensor of shape (*, m, n).
            B (torch.Tensor): Input tensor of shape (*, k, n).

        Returns:
            torch.Tensor: Output tensor after performing torch.matmul(A, B.T).
        """
        return torch.matmul(A, B.T)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_MatmulTransB.json")
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
