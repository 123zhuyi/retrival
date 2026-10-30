import torch
import torch.nn as nn
import math
import json
import os


class Model(nn.Module):
    """
    FlexAttention Forward (small-op composition, no flex_attention dependency).

    Implements the flex_attention semantics subset covered by the bench cases
    (score_mod=None, causal mask_mod, optional GQA) by composing basic
    operators, matching the math of the official kernel:

        S = Q @ K^T * scale
        if causal: S = masked_fill(S, -inf)
        P = exp(S - logsumexp(S))
        O = P @ V

    GQA: KV heads are expanded to Q heads before the matmuls, equivalent to
    enable_gqa=True in the official API (requires H_q % H_kv == 0).

    Computation is done in fp32 for numerical stability, output is cast back
    to the input dtype. Note: this reference materializes the S_q x S_k score
    matrix; the official fused kernel never does.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, query, key, value, is_causal=False, enable_gqa=False):
        B, H, S_q, D = query.shape
        H_kv = key.shape[1]
        S_k = key.shape[2]
        scale = 1.0 / math.sqrt(D)

        query_f = query.float()
        key_f = key.float()
        value_f = value.float()

        # GQA: expand KV heads to Q heads
        if enable_gqa and H != H_kv:
            rep = H // H_kv
            key_f = key_f.repeat_interleave(rep, dim=1)
            value_f = value_f.repeat_interleave(rep, dim=1)

        scores = torch.matmul(query_f, key_f.transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(S_q, S_k, device=query.device, dtype=torch.bool),
                diagonal=S_k - S_q + 1)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        logsumexp = torch.logsumexp(scores, dim=-1)
        attn_probs = torch.exp(scores - logsumexp.unsqueeze(-1))
        attn_output = torch.matmul(attn_probs, value_f)

        return attn_output.to(query.dtype)


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "103_FlexAttentionFwd.json")
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

        q_info = next(inp for inp in inputs if inp["name"] == "query")
        k_info = next(inp for inp in inputs if inp["name"] == "key")
        v_info = next(inp for inp in inputs if inp["name"] == "value")
        dtype = dtype_map[q_info["dtype"]]

        query = random_tensor(q_info["shape"], dtype)
        key = random_tensor(k_info["shape"], dtype)
        value = random_tensor(v_info["shape"], dtype) * 0.2
        is_causal = next((inp["value"] for inp in inputs if inp["name"] == "is_causal"), False)
        enable_gqa = next((inp["value"] for inp in inputs if inp["name"] == "enable_gqa"), False)

        input_groups.append([query, key, value, is_causal, enable_gqa])
    return input_groups


def get_init_inputs():
    return []