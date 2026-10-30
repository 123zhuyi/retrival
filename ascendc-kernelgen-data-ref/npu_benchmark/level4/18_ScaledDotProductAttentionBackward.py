import torch
import math
import json
import os

import torch.nn as nn


# ---------------------------------------------------------------------------
# ScaledDotProductAttentionBackward —— 融合 SDPA 反向（flash_attention_score_grad 算法）
#
# 借鉴 ops-ref `flash_attention_score_grad` AscendC kernel 的融合反向算法：
#   已知前向  S = QK^T·scale（causal 时 mask）； P = Softmax(S)； O = P·V
#   反向（前向保存 logsumexp，重建 softmax，无需重跑前向 softmax）：
#             dV = P^T·dO
#             dP = dO·V^T
#             D  = rowsum(dO ⊙ O)            （kernel 的 softmax_grad 项）
#             dS = P ⊙ (dP - D)
#             dQ = dS·K·scale； dK = dS^T·Q·scale
#
# 融合要点（对齐 kernel 实现）：
#   1. 全程 fp32 计算（kernel 内部 upcast 精度），输出回铸输入 dtype。
#   2. causal 掩码用 torch.triu 一次生成（kernel sparse_mode=causal）。
#   3. 直接由 logsumexp 重建 P = exp(S - logsumexp)，等价 kernel 的
#      online softmax 反推（用保存的 LSE，不做完整前向）。
#   4. scale 保留在 QK^T 之后乘（与 29 同理，长序列 bf16 数值稳定）。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """
    Scaled Dot-Product Attention Backward (fused flash_attention_score_grad style).

    Computes gradients w.r.t. query, key, and value using the logsumexp
    from forward pass for numerical stability. Supports optional causal masking.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, grad_output, query, key, value, attn_output, logsumexp, is_causal=False):
        B, H, S_q, D = query.shape
        S_k = key.shape[2]
        scale = 1.0 / math.sqrt(D)
        dtype = query.dtype
        query_f = query.float()
        key_f = key.float()
        value_f = value.float()
        grad_output_f = grad_output.float()

        # 重建注意力分数 S = QK^T·scale（causal mask 一次生成，kernel sparse_mode）
        attn_weight = torch.matmul(query_f, key_f.transpose(-2, -1)) * scale
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(S_q, S_k, device=query.device, dtype=torch.bool),
                diagonal=S_k - S_q + 1)
            attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))

        # 由前向保存的 logsumexp 重建 P = exp(S - logsumexp)
        attn_probs = torch.exp(attn_weight - logsumexp.unsqueeze(-1).float())

        # ---- 反向（kernel 公式）：dV/dP/D/dS/dQ/dK ----
        grad_value = torch.matmul(attn_probs.transpose(-2, -1), grad_output_f)  # dV = P^T·dO
        grad_attn_probs = torch.matmul(grad_output_f, value_f.transpose(-2, -1))  # dP = dO·V^T
        D_term = (grad_output_f * attn_output.float()).sum(dim=-1, keepdim=True)  # D = rowsum(dO⊙O)
        grad_attn_weight = attn_probs * (grad_attn_probs - D_term)  # dS = P⊙(dP - D)

        grad_query = torch.matmul(grad_attn_weight, key_f) * scale  # dQ = dS·K·scale
        grad_key = torch.matmul(grad_attn_weight.transpose(-2, -1), query_f) * scale  # dK = dS^T·Q·scale

        return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "18_ScaledDotProductAttentionBackward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype_, zero_mean=False):
        mu = 0.0 if zero_mean else float(torch.empty(1).uniform_(0.5, 1.2).item())
        sigma = float(torch.empty(1).uniform_(0.5, 1.5).item())
        if dtype_ is torch.bfloat16:
            return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype_)
        return torch.normal(mu, sigma, shape, dtype=dtype_)

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
        key = random_tensor(k_info["shape"], dtype, zero_mean=True)
        value = random_tensor(v_info["shape"], dtype) * 0.2
        is_causal = next(inp for inp in inputs if inp["name"] == "is_causal")["value"]
        with torch.no_grad():
            q_f = query.float()
            k_f = key.float()
            v_f = value.float()
            B, H, S_q, D = q_f.shape
            S_k = k_f.shape[2]
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(S_q, S_k, device=q_f.device, dtype=torch.bool),
                    diagonal=S_k - S_q + 1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            logsumexp = torch.logsumexp(scores, dim=-1)
            attn_probs = torch.exp(scores - logsumexp.unsqueeze(-1))
            attn_output = torch.matmul(attn_probs, v_f).to(dtype)

        input_groups.append([grad_output, query, key, value, attn_output, logsumexp, is_causal])
    return input_groups


def get_init_inputs():
    return []
