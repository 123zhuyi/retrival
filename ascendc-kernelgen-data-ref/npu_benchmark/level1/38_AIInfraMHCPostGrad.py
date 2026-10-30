import torch
import torch.nn as nn
import json
import os
import omni_training_custom_ops


class Model(nn.Module):
    r"""
    torch.ops.custom.npu_ai_infra_mhc_post_grad = npu_ai_infra_mhc_post_grad(...)
    npu_ai_infra_mhc_post_grad(grad_output, x, h_res, h_out, h_post) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    Computes gradients for the manifold constrained hyper connection post-processing operation.

    Args:
        grad_output (Tensor): Upstream gradient of shape [B, S, n, D].
        x (Tensor): Input tensor of shape [B, S, n, D].
        h_res (Tensor): Residual tensor of shape [B, S, n, n].
        h_out (Tensor): Output tensor of shape [B, S, D].
        h_post (Tensor): Post tensor of shape [B, S, n].

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: grad_x, grad_h_res, grad_h_out, grad_h_post.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, x: torch.Tensor,
                h_res: torch.Tensor, h_out: torch.Tensor,
                h_post: torch.Tensor) -> tuple:
        """
        Computes gradients for manifold constrained hyper connection post-processing.

        Args:
            grad_output (Tensor): Upstream gradient.
            x (Tensor): Input tensor.
            h_res (Tensor): Residual tensor.
            h_out (Tensor): Output tensor.
            h_post (Tensor): Post tensor.

        Returns:
            Tuple[Tensor, ...]: Gradients.
        """
        return torch.ops.custom.npu_ai_infra_mhc_post_grad(
            grad_output, x, h_res, h_out, h_post
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "38_AIInfraMHCPostGrad.json")
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
        x = make_tensor(inputs[1])
        h_res = make_tensor(inputs[2])
        h_out = make_tensor(inputs[3])
        h_post = make_tensor(inputs[4])
        input_groups.append([grad_output, x, h_res, h_out, h_post])
    return input_groups


def get_init_inputs():
    return []
