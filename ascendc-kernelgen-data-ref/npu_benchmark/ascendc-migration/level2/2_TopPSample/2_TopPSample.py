import torch
import torch.nn as nn
import torch_npu
import json
import os


class Model(nn.Module):
    r"""
    torch_npu.npu_top_k_top_p_sample = npu_top_k_top_p_sample(logits, top_k, top_p, q=None, eps=1e-8, is_need_logits=False, top_k_guess=32) -> (Tensor, Tensor)

    Top-P (nucleus) sampling. This wrapper configures the fused operator for pure topP mode.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def top_p_sample_native(logits, top_p, eps=1e-8):
            # 1. Softmax
            probs = torch.softmax(logits, dim=-1)

            # 2. Sort descending
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

            # 3. Cumsum
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # 4. Top-P mask: remove tokens where cumsum > top_p
            top_p_mask = cumsum_probs > top_p.unsqueeze(-1)
            top_p_mask[..., 0] = False  # keep at least one token

            # 5. Apply mask to sorted logits
            sorted_logits = logits.gather(-1, sorted_indices)
            filtered_logits = sorted_logits.masked_fill(top_p_mask, float('-inf'))

            # 6. Sample from filtered distribution
            filtered_probs = torch.softmax(filtered_logits, dim=-1)
            sampled_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)

            return sampled_idx, filtered_logits

    Args:
        logits (Tensor): Input logits tensor of shape [batch, vocab_size].
            Supports fp16, bf16, fp32.
        top_p (Tensor or float): Top-P threshold. If tensor, shape [batch].
        eps (float, optional): Epsilon for numerical stability. Default: 1e-8.

    Returns:
        tuple: (sampled_indices, filtered_logits)

    Example::

        >>> logits = torch.randn(2, 32000, dtype=torch.float16)
        >>> top_p = torch.tensor([0.9, 0.9], dtype=torch.float16)
        >>> torch_npu.npu_top_k_top_p_sample(
        ...     logits,
        ...     torch.ones(2, 1, dtype=torch.int32),  # invalid top_k to skip
        ...     top_p,
        ...     q=None,
        ...     eps=1e-8
        ... )
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, logits: torch.Tensor, top_p, eps: float = 1e-8) -> tuple:
        """
        Performs top-p (nucleus) sampling.

        Args:
            logits (torch.Tensor): Input logits [batch, vocab_size].
            top_p: Top-P threshold (float or [batch] tensor).
            eps (float, optional): Epsilon. Default: 1e-8.

        Returns:
            tuple: (sampled_indices, filtered_logits).
        """
        batch_size = logits.shape[0]
        device = logits.device

        # Pass invalid top_k values to skip topK filtering
        top_k = torch.ones(batch_size, 1, dtype=torch.int32, device=device)

        # q=None: skip exponential sampling
        q = None

        return torch_npu.npu_top_k_top_p_sample(
            logits, top_k, top_p, q=q, eps=eps, is_need_logits=False, top_k_guess=32
        )


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "2_TopPSample.json")

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

        logits_info = inputs[0]
        top_p_info = inputs[1]
        eps_info = inputs[2] if len(inputs) > 2 else {"value": 1e-8}

        logits_dtype = dtype_map[logits_info["dtype"]]
        logits_shape = logits_info["shape"]
        logits_fill = logits_info.get("fill", None)

        if logits_fill in fill_map:
            logits = torch.full(logits_shape, fill_map[logits_fill], dtype=logits_dtype)
        elif idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            raw = torch.normal(mu, sigma, logits_shape, dtype=torch.float32)
            logits = raw.to(logits_dtype)
        else:
            raw = torch.empty(logits_shape, dtype=torch.float32).uniform_(-5, 5)
            logits = raw.to(logits_dtype)

        # top_p: can be scalar value or tensor
        if "dtype" in top_p_info:
            top_p_dtype = dtype_map[top_p_info["dtype"]]
            top_p_shape = top_p_info["shape"]
            top_p_fill = top_p_info.get("fill", None)
            if top_p_fill in fill_map:
                top_p = torch.full(top_p_shape, fill_map[top_p_fill], dtype=top_p_dtype)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, top_p_shape, dtype=torch.float32)
                top_p = raw.to(top_p_dtype)
            else:
                raw = torch.empty(top_p_shape, dtype=torch.float32).uniform_(-5, 5)
                top_p = raw.to(top_p_dtype)
        else:
            top_p = top_p_info["value"]

        eps = eps_info["value"]

        input_groups.append([logits, top_p, eps])

    return input_groups


def get_init_inputs():
    return []