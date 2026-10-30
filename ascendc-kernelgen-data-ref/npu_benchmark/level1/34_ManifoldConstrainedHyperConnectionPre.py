import torch
import torch.nn as nn
import json
import os
import omni_training_custom_ops


class Model(nn.Module):
    r"""
    torch.ops.custom.npu_manifold_constrained_hyper_connection_pre = npu_manifold_constrained_hyper_connection_pre(...)
    npu_manifold_constrained_hyper_connection_pre(x, phi, alpha, bias, *, gamma, out_flag) -> Tensor

    Performs the manifold constrained hyper connection pre-processing operation.

    Args:
        x (Tensor): Input tensor of shape [T, n, D].
        phi (Tensor): Weight tensor of shape [n*n + 2*n, n*D].
        alpha (Tensor): Scale tensor of shape [3].
        bias (Tensor): Bias tensor of shape [n*n + 2*n].
        gamma (Tensor): Gamma tensor of shape [n, D].
        out_flag (int, optional): Output flag. Default: 1.

    Returns:
        Tensor: Output tensor.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, phi: torch.Tensor, alpha: torch.Tensor,
                bias: torch.Tensor, gamma: torch.Tensor, out_flag: int = 1) -> torch.Tensor:
        """
        Applies manifold constrained hyper connection pre-processing.

        Args:
            x (Tensor): Input tensor of shape [T, n, D].
            phi (Tensor): Weight tensor of shape [n*n + 2*n, n*D].
            alpha (Tensor): Scale tensor of shape [3].
            bias (Tensor): Bias tensor of shape [n*n + 2*n].
            gamma (Tensor): Gamma tensor of shape [n, D].
            out_flag (int, optional): Output flag.

        Returns:
            Tensor: Output tensor.
        """
        return torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(
            x, phi, alpha, bias, gamma=gamma, out_flag=out_flag
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "34_ManifoldConstrainedHyperConnectionPre.json")
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

        x = make_tensor(inputs[0])
        phi = make_tensor(inputs[1])
        alpha = make_tensor(inputs[2])
        bias = make_tensor(inputs[3])
        gamma = make_tensor(inputs[4])
        out_flag = inputs[5]["value"]
        input_groups.append([x, phi, alpha, bias, gamma, out_flag])
    return input_groups


def get_init_inputs():
    return []
