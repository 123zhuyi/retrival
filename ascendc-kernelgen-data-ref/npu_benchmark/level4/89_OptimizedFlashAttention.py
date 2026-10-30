import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Flash Attention with RoPE / ALiBi, accepting Q/K/V in BNSD layout.
    """

    def __init__(self):
        super(Model, self).__init__()

    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor, d_k: int) -> torch.Tensor:
        """Apply Rotary Position Embedding (RoPE) to x (BNSD)."""
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_k, 2, device=x.device).float() / d_k))
        sincos = torch.einsum('bi,j->bij', position_ids.float(), inv_freq)
        sin = sincos.sin().repeat_interleave(2, dim=-1)
        cos = sincos.cos().repeat_interleave(2, dim=-1)
        x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)
        return x * cos.unsqueeze(1) + x_rot * sin.unsqueeze(1)

    def _compute_alibi_bias(self, seq_len_q: int, seq_len_k: int, n_heads: int, device: torch.device) -> torch.Tensor:
        """Compute ALiBi bias for BNSD attention."""
        slopes = 2 ** (-8 * torch.arange(n_heads, device=device) / n_heads)
        q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1)
        k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
        relative_pos = -(q_pos - k_pos).abs()
        return slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos.unsqueeze(0)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                use_rope: bool = True, use_alibi: bool = False,
                scale: float = None, pse: torch.Tensor = None,
                sink: torch.Tensor = None) -> torch.Tensor:
        """
        Flash Attention with optional RoPE / ALiBi.
        query/key/value are expected in BNSD layout, matching the original benchmark interface.
        """
        torch.manual_seed(42)
        b, n, sq, d_k = query.shape
        _, _, skv, _ = key.shape
        _, _, _, dv = value.shape

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scale = scale or (1.0 / (d_k ** 0.5))
        scores = scores * scale

        # RoPE (applied after projection, before softmax)
        if use_rope:
            position_ids = torch.arange(sq, device=query.device).unsqueeze(0).expand(b, -1)
            query = self._apply_rope(query, position_ids, d_k)
            key = self._apply_rope(key, position_ids, d_k)
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # ALiBi
        if use_alibi:
            alibi_bias = self._compute_alibi_bias(sq, skv, n, query.device)
            scores = scores + alibi_bias.unsqueeze(0)

        # Optional pse / sink
        if pse is not None:
            pse = pse.to(scores.dtype)
            if pse.dim() == 4:
                scores = scores + pse
            elif pse.dim() == 2:
                scores = scores + pse.view(1, 1, 1, -1)
            else:
                scores = scores + pse

        if sink is not None:
            sink = sink.to(scores.dtype)
            scores = scores + sink.view(1, -1, 1, 1)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "89_OptimizedFlashAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        # JSON 中实际字段为 x: [batch, seq_len, d_model]，拆分为 Q/K/V (BNSD)
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = dtype_map[x_info["dtype"]]
        x = random_tensor(x_info["shape"], dtype)

        batch_size, seq_len, d_model = x.shape
        # infer n_heads as a divisor of d_model, prefer 8
        n_heads = 8
        for h in [8, 4, 2, 1]:
            if d_model % h == 0:
                n_heads = h
                break
        d_k = d_model // n_heads

        # Q/K/V all share x (self-attention), shape [B, N, S, D]
        query = x.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
        key = query.clone()
        value = query.clone()

        # optional parameters
        use_rope = True
        use_alibi = False
        scale = None
        pse = None
        sink = None

        for inp in inputs[1:]:
            name = inp.get("name", "")
            if name == "use_rope":
                use_rope = inp["value"]
            elif name == "use_alibi":
                use_alibi = inp["value"]
            elif name == "scale":
                scale = inp["value"]
            elif name == "pse":
                pse = random_tensor(inp["shape"], dtype)
            elif name == "sink":
                sink = random_tensor(inp["shape"], torch.float32)

        input_groups.append([query, key, value, use_rope, use_alibi, scale, pse, sink])
    return input_groups


def get_init_inputs():
    return []