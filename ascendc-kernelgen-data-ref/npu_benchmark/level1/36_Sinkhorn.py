import torch
import torch.nn as nn
import json
import os
import omni_training_custom_ops


class Model(nn.Module):
    r"""
    torch.ops.custom.npu_sinkhorn = npu_sinkhorn(...)
    npu_sinkhorn(x, *, out_flag, eps, num_iters) -> (Tensor, Tensor, Tensor)

    Performs the Sinkhorn normalization operation.

    Args:
        x (Tensor): Input tensor of shape [..., n, n].
        out_flag (int, optional): Output flag. Default: 1.
        eps (float, optional): Epsilon for numerical stability. Default: 1e-8.
        num_iters (int, optional): Number of iterations. Default: 20.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Output tensor, norm output, and sum output.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, out_flag: int = 1,
                eps: float = 1e-8, num_iters: int = 20) -> tuple:
        """
        Applies Sinkhorn normalization.

        Args:
            x (Tensor): Input tensor.
            out_flag (int, optional): Output flag.
            eps (float, optional): Epsilon for numerical stability.
            num_iters (int, optional): Number of iterations.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Output tensor, norm output, and sum output.
        """
        return torch.ops.custom.npu_sinkhorn(
            x, out_flag=out_flag, eps=float(eps), num_iters=int(num_iters)
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "36_Sinkhorn.json")
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
        out_flag = inputs[1]["value"]
        eps = inputs[2]["value"]
        num_iters = inputs[3]["value"]
        input_groups.append([x, out_flag, eps, num_iters])
    return input_groups


def get_init_inputs():
    return []
