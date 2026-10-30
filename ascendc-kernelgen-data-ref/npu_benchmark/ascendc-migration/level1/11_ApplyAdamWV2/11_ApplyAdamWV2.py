import torch
import torch.nn as nn
import json
import os
import math


class Model(nn.Module):
    r"""
    aclnnApplyAdamWV2

    Fused AdamW optimizer step.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def apply_adam_w_v2_native(param, exp_avg, exp_avg_sq, max_exp_avg_sq,
                                   grad, step, lr, beta1, beta2, weight_decay,
                                   eps, amsgrad, maximize):
            # dtype alignment
            dtype = param.dtype
            if grad.dtype != dtype:
                grad = grad.to(dtype)
                max_exp_avg_sq = max_exp_avg_sq.to(dtype)
            if maximize:
                grad = -grad

            step = step.item() + 1

            # weight decay
            param = param * (1 - lr * weight_decay)

            # momentum update
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # bias correction
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)

            if amsgrad:
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

            param.addcdiv_(exp_avg, denom, value=-step_size)

            return param, exp_avg, exp_avg_sq, max_exp_avg_sq

    Args:
        param (Tensor): Model parameter to update.
        exp_avg (Tensor): Exponential moving average of gradients.
        exp_avg_sq (Tensor): Exponential moving average of squared gradients.
        max_exp_avg_sq (Tensor): Max of exp_avg_sq for AMSGrad.
        grad (Tensor): Gradient.
        step (Tensor): Current step, scalar int64 or float32.
        lr (float): Learning rate.
        beta1 (float): Coefficient for running average of gradient.
        beta2 (float): Coefficient for running average of squared gradient.
        weight_decay (float): Weight decay coefficient.
        eps (float): Term added for numerical stability.
        amsgrad (bool): Whether to use AMSGrad variant.
        maximize (bool): Whether to maximize rather than minimize.

    Returns:
        tuple: (updated_param, updated_exp_avg, updated_exp_avg_sq, updated_max_exp_avg_sq)

    Example::

        >>> param = torch.randn(4, 4)
        >>> exp_avg = torch.zeros_like(param)
        >>> exp_avg_sq = torch.zeros_like(param)
        >>> max_exp_avg_sq = torch.zeros_like(param)
        >>> grad = torch.randn(4, 4)
        >>> step = torch.tensor([1])
        >>> apply_adam_w_v2_native(param, exp_avg, exp_avg_sq, max_exp_avg_sq,
        ...                        grad, step, 0.001, 0.9, 0.999, 0.01, 1e-8, False, False)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, param, exp_avg, exp_avg_sq, max_exp_avg_sq, grad, step,
                lr: float, beta1: float, beta2: float, weight_decay: float,
                eps: float, amsgrad: bool, maximize: bool):
        """
        AdamW optimizer step.

        Args:
            param (torch.Tensor): Parameter tensor.
            exp_avg (torch.Tensor): First moment estimate.
            exp_avg_sq (torch.Tensor): Second moment estimate.
            max_exp_avg_sq (torch.Tensor): Max second moment for AMSGrad.
            grad (torch.Tensor): Gradient tensor.
            step (torch.Tensor): Step counter.
            lr (float): Learning rate.
            beta1 (float): Beta1.
            beta2 (float): Beta2.
            weight_decay (float): Weight decay.
            eps (float): Epsilon.
            amsgrad (bool): AMSGrad flag.
            maximize (bool): Maximize flag.

        Returns:
            tuple: Updated (param, exp_avg, exp_avg_sq, max_exp_avg_sq).
        """
        # dtype alignment
        dtype = param.dtype
        if grad.dtype != dtype:
            grad = grad.to(dtype)
            max_exp_avg_sq = max_exp_avg_sq.to(dtype)
        if maximize:
            grad = -grad

        step_val = int(step.item()) + 1

        # weight decay
        param = param * (1 - lr * weight_decay)

        # momentum update
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # bias correction
        bias_correction1 = 1 - beta1 ** step_val
        bias_correction2 = 1 - beta2 ** step_val
        step_size = lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        if amsgrad:
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
        else:
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps

        param.addcdiv_(exp_avg, denom, value=-step_size)

        return param, exp_avg, exp_avg_sq, max_exp_avg_sq


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "11_ApplyAdamWV2.json")

    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    fill_map = {
        'inf': float('inf'),
        '-inf': float('-inf'),
        'nan': float('nan'),
        '-nan': -float('nan'),
        'zero': 0.0,
    }

    input_groups = []

    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        scene = case.get("scene", "")

        param_info = inputs[0]
        exp_avg_info = inputs[1]
        exp_avg_sq_info = inputs[2]
        max_exp_avg_sq_info = inputs[3]
        grad_info = inputs[4]
        step_info = inputs[5]
        lr_info = inputs[6]
        beta1_info = inputs[7]
        beta2_info = inputs[8]
        weight_decay_info = inputs[9]
        eps_info = inputs[10]
        amsgrad_info = inputs[11]
        maximize_info = inputs[12]

        def make_tensor(info, idx):
            dtype = dtype_map[info["dtype"]]
            shape = info["shape"]
            fill = info.get("fill", None)
            if fill in fill_map:
                return torch.full(shape, fill_map[fill], dtype=dtype)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, shape, dtype=torch.float32)
                return raw.to(dtype)
            else:
                raw = torch.empty(shape, dtype=torch.float32).uniform_(-5, 5)
                return raw.to(dtype)

        param = make_tensor(param_info, idx)
        exp_avg = make_tensor(exp_avg_info, idx)
        exp_avg_sq = make_tensor(exp_avg_sq_info, idx)
        max_exp_avg_sq = make_tensor(max_exp_avg_sq_info, idx)
        grad = make_tensor(grad_info, idx)
        step = make_tensor(step_info, idx)

        lr = lr_info["value"]
        beta1 = beta1_info["value"]
        beta2 = beta2_info["value"]
        weight_decay = weight_decay_info["value"]
        eps = eps_info["value"]
        amsgrad = amsgrad_info["value"]
        maximize = maximize_info["value"]

        input_groups.append([
            param, exp_avg, exp_avg_sq, max_exp_avg_sq, grad, step,
            lr, beta1, beta2, weight_decay, eps, amsgrad, maximize
        ])

    return input_groups


def get_init_inputs():
    return []