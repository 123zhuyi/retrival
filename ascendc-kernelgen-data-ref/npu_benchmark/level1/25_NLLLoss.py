import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
        Simple model that computes the negative log likelihood loss.
        torch.nn.functional.nll_loss = nll_loss(
        input: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = 'mean'
    ) -> Tensor
        Compute the negative log likelihood loss.

        See :class:`~torch.nn.NLLLoss` for details.

        Args:
            input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
                in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
                in the case of K-dimensional loss. `input` is expected to be log-probabilities.
            target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
                or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
                K-dimensional loss.
            weight (Tensor, optional): A manual rescaling weight given to each
                class. If given, has to be a Tensor of size `C`
            size_average (bool, optional): Deprecated (see :attr:`reduction`).
            ignore_index (int, optional): Specifies a target value that is ignored
                and does not contribute to the input gradient. When :attr:`size_average` is
                ``True``, the loss is averaged over non-ignored targets. Default: -100
            reduce (bool, optional): Deprecated (see :attr:`reduction`).
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        Example::

            >>> # input is of size N x C = 3 x 5
            >>> input = torch.randn(3, 5, requires_grad=True)
            >>> # each element in target has to have 0 <= value < C
            >>> target = torch.tensor([1, 0, 4])
            >>> output = F.nll_loss(F.log_softmax(input, dim=1), target)
            >>> output.backward()
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor,
                weight: torch.Tensor = None, ignore_index: int = -100,
                reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the negative log likelihood loss.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C) or (N, C, d1, d2, ...).
            target (torch.Tensor): Target tensor of shape (N,) or (N, d1, d2, ...).
            weight (torch.Tensor, optional): Manual rescaling weight given to each class.
            ignore_index (int, optional): Target value that is ignored and does not contribute to gradient.
            reduction (str, optional): Reduction method ('none', 'mean', 'sum').

        Returns:
            torch.Tensor: NLL loss value.
        """
        return torch.nn.functional.nll_loss(input, target, weight=weight, ignore_index=ignore_index, reduction=reduction)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "25_NLLLoss.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        input_info = inputs[0]
        target_info = inputs[1]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[input_info["dtype"]]
        
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5, 0).item())
            sigma = float(torch.empty(1).uniform_(0.3, 1.5).item())
            input_tensor = torch.normal(mu, sigma, input_info["shape"], dtype=dtype)
        else:
            input_tensor = torch.empty(input_info["shape"], dtype=dtype).uniform_(-5, 0)
        
        target_range = target_info.get("range", [0, input_info["shape"][1] - 1])
        target = torch.randint(target_range[0], target_range[1] + 1, tuple(target_info["shape"]), dtype=torch.int64)
        
        weight = None
        ignore_index = -100
        reduction = "mean"
        
        for inp in inputs[2:]:
            if inp["name"] == "weight":
                weight = torch.randn(inp["shape"], dtype=dtype)
            elif inp["name"] == "ignore_index":
                ignore_index = inp["value"]
            elif inp["name"] == "reduction":
                reduction = inp["value"]
        
        input_groups.append([input_tensor, target, weight, ignore_index, reduction])
    return input_groups


def get_init_inputs():
    return []
