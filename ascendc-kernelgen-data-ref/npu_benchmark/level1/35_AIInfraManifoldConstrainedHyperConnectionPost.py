import torch
import torch.nn as nn
import json
import os
import omni_training_custom_ops


class Model(nn.Module):
    r"""
    torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post = npu_ai_infra_manifold_constrained_hyper_connection_post(...)
    npu_ai_infra_manifold_constrained_hyper_connection_post(x, h_res, h_out, h_post) -> Tensor

    Performs the manifold constrained hyper connection post-processing operation.

    Args:
        x (Tensor): Input tensor of shape [B, S, n, D].
        h_res (Tensor): Residual tensor of shape [B, S, n, n].
        h_out (Tensor): Output tensor of shape [B, S, D].
        h_post (Tensor): Post tensor of shape [B, S, n].

    Returns:
        Tensor: Output tensor.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, h_res: torch.Tensor, h_out: torch.Tensor,
                h_post: torch.Tensor) -> torch.Tensor:
        """
        Applies manifold constrained hyper connection post-processing.

        Args:
            x (Tensor): Input tensor of shape [B, S, n, D].
            h_res (Tensor): Residual tensor of shape [B, S, n, n].
            h_out (Tensor): Output tensor of shape [B, S, D].
            h_post (Tensor): Post tensor of shape [B, S, n].

        Returns:
            Tensor: Output tensor.
        """
        return torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
            x, h_res, h_out, h_post
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "35_AIInfraManifoldConstrainedHyperConnectionPost.json")
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
        h_res = make_tensor(inputs[1])
        h_out = make_tensor(inputs[2])
        h_post = make_tensor(inputs[3])
        input_groups.append([x, h_res, h_out, h_post])
    return input_groups


def get_init_inputs():
    return []
