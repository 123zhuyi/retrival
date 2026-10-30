import torch
import torch.nn as nn
import json
import os
import omni_training_custom_ops


class Model(nn.Module):
    r"""
    torch.ops.custom.npu_manifold_constrained_hyper_connection_pre_grad = npu_manifold_constrained_hyper_connection_pre_grad(...)
    npu_manifold_constrained_hyper_connection_pre_grad(
        x, phi, alpha, h_in_grad, h_post_grad, h_comb_before_grad,
        inv_rms, mm_res, h_pre, h_post, *, gamma, hc_eps, grad_x_post
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    Computes gradients for the manifold constrained hyper connection pre-processing operation.

    Args:
        x (Tensor): Input tensor of shape [B, S, n, D].
        phi (Tensor): Weight tensor of shape [n*n + 2*n, n*D].
        alpha (Tensor): Scale tensor of shape [3].
        h_in_grad (Tensor): Gradient of h_in of shape [B, S, D].
        h_post_grad (Tensor): Gradient of h_post of shape [B, S, n].
        h_comb_before_grad (Tensor): Gradient of h_comb_before of shape [B, S, n, n].
        inv_rms (Tensor): Inverse RMS tensor of shape [B, S].
        mm_res (Tensor): Matrix result tensor of shape [B, S, n*n + 2*n].
        h_pre (Tensor): h_pre tensor of shape [B, S, n].
        h_post (Tensor): h_post tensor of shape [B, S, n].
        gamma (Tensor): Gamma tensor of shape [n, D].
        hc_eps (float, optional): Epsilon. Default: 1e-6.
        grad_x_post (Tensor): Gradient of x_post of shape [B, S, n, D].

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: grad_x, grad_phi, grad_alpha, grad_bias, grad_gamma.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, phi: torch.Tensor, alpha: torch.Tensor,
                h_in_grad: torch.Tensor, h_post_grad: torch.Tensor,
                h_comb_before_grad: torch.Tensor, inv_rms: torch.Tensor,
                mm_res: torch.Tensor, h_pre: torch.Tensor, h_post: torch.Tensor,
                gamma: torch.Tensor, grad_x_post: torch.Tensor,
                hc_eps: float = 1e-6) -> tuple:
        """
        Computes gradients for manifold constrained hyper connection pre-processing.

        Args:
            x (Tensor): Input tensor.
            phi (Tensor): Weight tensor.
            alpha (Tensor): Scale tensor.
            h_in_grad (Tensor): Gradient of h_in.
            h_post_grad (Tensor): Gradient of h_post.
            h_comb_before_grad (Tensor): Gradient of h_comb_before.
            inv_rms (Tensor): Inverse RMS tensor.
            mm_res (Tensor): Matrix result tensor.
            h_pre (Tensor): h_pre tensor.
            h_post (Tensor): h_post tensor.
            gamma (Tensor): Gamma tensor.
            grad_x_post (Tensor): Gradient of x_post.
            hc_eps (float, optional): Epsilon.

        Returns:
            Tuple[Tensor, ...]: Gradients.
        """
        return torch.ops.custom.npu_manifold_constrained_hyper_connection_pre_grad(
            x, phi, alpha, h_in_grad, h_post_grad, h_comb_before_grad,
            inv_rms, mm_res, h_pre, h_post,
            gamma=gamma, hc_eps=hc_eps, grad_x_post=grad_x_post
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "37_ManifoldConstrainedHyperConnectionPreGrad.json")
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
        h_in_grad = make_tensor(inputs[3])
        h_post_grad = make_tensor(inputs[4])
        h_comb_before_grad = make_tensor(inputs[5])
        inv_rms = make_tensor(inputs[6])
        mm_res = make_tensor(inputs[7])
        h_pre = make_tensor(inputs[8])
        h_post = make_tensor(inputs[9])
        gamma = make_tensor(inputs[10])
        grad_x_post = make_tensor(inputs[11])
        hc_eps = inputs[12]["value"]
        input_groups.append([
            x, phi, alpha, h_in_grad, h_post_grad, h_comb_before_grad,
            inv_rms, mm_res, h_pre, h_post, gamma, grad_x_post, hc_eps
        ])
    return input_groups


def get_init_inputs():
    return []
