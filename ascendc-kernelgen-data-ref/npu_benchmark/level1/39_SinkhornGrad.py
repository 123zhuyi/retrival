import torch
import torch.nn as nn
import json
import os
import omni_training_custom_ops


class Model(nn.Module):
    r"""
    torch.ops.custom.npu_sinkhorn_grad = npu_sinkhorn_grad(...)
    npu_sinkhorn_grad(grad_output, norm_out, sum_out) -> Tensor

    Computes the gradient for the Sinkhorn normalization operation.

    Args:
        grad_output (Tensor): Upstream gradient of shape [T, n, n].
        norm_out (Tensor): Norm output of shape [2*num_iters, n, n, T].
        sum_out (Tensor): Sum output of shape [2*num_iters, n, T].

    Returns:
        Tensor: Gradient input tensor.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, norm_out: torch.Tensor,
                sum_out: torch.Tensor) -> torch.Tensor:
        """
        Computes gradient for Sinkhorn normalization.

        Args:
            grad_output (Tensor): Upstream gradient.
            norm_out (Tensor): Norm output.
            sum_out (Tensor): Sum output.

        Returns:
            Tensor: Gradient input tensor.
        """
        return torch.ops.custom.npu_sinkhorn_grad(grad_output, norm_out, sum_out)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "39_SinkhornGrad.json")
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

        def make_tensor(info):
            dtype = dtype_map[info["dtype"]]
            if idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                t = torch.normal(mu, sigma, info["shape"])
            else:
                t = torch.empty(info["shape"], dtype=torch.float32).uniform_(-5, 5)
            return t.to(dtype)

        grad_output = make_tensor(inputs[0])
        norm_out = make_tensor(inputs[1])
        sum_out = make_tensor(inputs[2])
        input_groups.append([grad_output, norm_out, sum_out])
    return input_groups


def get_init_inputs():
    return []
