import torch
import math
import json
import os


import torch.nn as nn


class Model(nn.Module):
    """
    FlexAttention Backward (template-style, pure backward logic).

    Computes gradients w.r.t. query, key, and value using the precomputed
    attn_output and logsumexp from the forward pass. This mirrors the
    gradient math of the official Inductor backward template in
    torch/_inductor/kernel/flex/flex_attention.py:

        dv = P^T @ dO
        dP = dO @ V^T
        D  = rowsum(dO * O)
        dS = P * (dP - D)
        dq = dS @ K * scale
        dk = dS^T @ Q * scale

    Supports optional causal masking and GQA (KV heads are expanded to Q
    heads; gradients are summed back to the original KV head count).

    Note: forward outputs (attn_output, logsumexp) are generated
    self-consistently in get_input_groups and passed in as inputs.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, grad_output, query, key, value, attn_output, logsumexp,
                is_causal=False, enable_gqa=False):
        B, H, S_q, D = query.shape
        H_kv = key.shape[1]
        S_k = key.shape[2]
        scale = 1.0 / math.sqrt(D)

        query_f = query.float()
        key_f = key.float()
        value_f = value.float()
        grad_output_f = grad_output.float()

        # GQA: expand KV heads to Q heads
        if enable_gqa and H != H_kv:
            rep = H // H_kv
            key_f = key_f.repeat_interleave(rep, dim=1)
            value_f = value_f.repeat_interleave(rep, dim=1)

        attn_weight = torch.matmul(query_f, key_f.transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(S_q, S_k, device=query.device, dtype=torch.bool),
                diagonal=S_k - S_q + 1)
            attn_weight.masked_fill_(causal_mask, float('-inf'))

        attn_probs = torch.exp(attn_weight - logsumexp.unsqueeze(-1).float())

        grad_value = torch.matmul(attn_probs.transpose(-2, -1), grad_output_f)
        grad_attn_probs = torch.matmul(grad_output_f, value_f.transpose(-2, -1))
        D_term = (grad_output_f * attn_output.float()).sum(dim=-1, keepdim=True)
        grad_attn_weight = attn_probs * (grad_attn_probs - D_term)

        grad_query = torch.matmul(grad_attn_weight, key_f) * scale
        grad_key = torch.matmul(grad_attn_weight.transpose(-2, -1), query_f) * scale

        # GQA: sum expanded-head grads back to KV heads
        if enable_gqa and H != H_kv:
            rep = H // H_kv
            grad_key = grad_key.view(B, H_kv, rep, S_k, D).sum(dim=2)
            grad_value = grad_value.view(B, H_kv, rep, S_k, D).sum(dim=2)

        return (grad_query.to(query.dtype),
                grad_key.to(key.dtype),
                grad_value.to(value.dtype))


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "104_FlexAttentionBwd.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype_):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype_ is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype_)
            return torch.normal(mu, sigma, shape, dtype=dtype_)
        else:
            return torch.empty(shape, dtype=dtype_).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

        grad_info = next(inp for inp in inputs if inp["name"] == "grad_output")
        q_info = next(inp for inp in inputs if inp["name"] == "query")
        k_info = next(inp for inp in inputs if inp["name"] == "key")
        v_info = next(inp for inp in inputs if inp["name"] == "value")
        dtype = dtype_map[q_info["dtype"]]

        grad_output = random_tensor(grad_info["shape"], dtype) * 0.2
        query = random_tensor(q_info["shape"], dtype)
        key = random_tensor(k_info["shape"], dtype)
        value = random_tensor(v_info["shape"], dtype) * 0.2
        is_causal = next((inp["value"] for inp in inputs if inp["name"] == "is_causal"), False)
        enable_gqa = next((inp["value"] for inp in inputs if inp["name"] == "enable_gqa"), False)

        # Compute logsumexp and true forward output from query/key/value
        # so that pre-given intermediates are self-consistent.
        with torch.no_grad():
            q_f = query.float()
            k_f = key.float()
            v_f = value.float()
            B, H, S_q, D = q_f.shape
            H_kv = k_f.shape[1]
            S_k = k_f.shape[2]
            scale = 1.0 / math.sqrt(D)

            if enable_gqa and H != H_kv:
                rep = H // H_kv
                k_f = k_f.repeat_interleave(rep, dim=1)
                v_f = v_f.repeat_interleave(rep, dim=1)

            scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(S_q, S_k, device=q_f.device, dtype=torch.bool),
                    diagonal=S_k - S_q + 1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            logsumexp = torch.logsumexp(scores, dim=-1)
            attn_probs = torch.exp(scores - logsumexp.unsqueeze(-1))
            attn_output = torch.matmul(attn_probs, v_f).to(dtype)

        input_groups.append([grad_output, query, key, value,
                             attn_output, logsumexp, is_causal, enable_gqa])
    return input_groups


def get_init_inputs():
    return []