import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    MLA (Multi-head Latent Attention) preprocessing operation.

    Performs linear projections (fused QKV-A or separate Q/KV projections),
    RMSNorm on compressed representations, and GPT-J style RoPE rotary
    position embeddings to prepare inputs for the attention kernel.
    Supports two paths: LoRA path (when q_lora_rank is not None) and
    direct projection path (when q_lora_rank is None).
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, hidden_states, positions, kv_lora_rank, qk_rope_head_dim,
                num_heads, qk_nope_head_dim, rope_base, q_lora_rank=None):
        torch.manual_seed(42)
        num_tokens, hidden_size = hidden_states.shape
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        key_biz = (hidden_size, kv_lora_rank, qk_rope_head_dim, num_heads,
                   qk_nope_head_dim, q_lora_rank)
        cache_key = (key_biz, device, dtype)
        if cache_key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key_biz) & 0xFFFFFFFF)

            out_channels = (kv_lora_rank + qk_rope_head_dim +
                           (q_lora_rank if q_lora_rank else 0))
            w_qkv_a = nn.Parameter(torch.empty(out_channels, hidden_size, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(w_qkv_a, a=5**0.5)

            if q_lora_rank is not None:
                w_q_b = nn.Parameter(
                    torch.empty(qk_head_dim * num_heads, q_lora_rank, dtype=dtype, device=device))
                nn.init.kaiming_uniform_(w_q_b, a=5**0.5)
                w_q_a_ln = nn.Parameter(torch.ones(q_lora_rank, dtype=dtype, device=device))
            else:
                w_q_b = nn.Parameter(
                    torch.empty(qk_head_dim * num_heads, hidden_size, dtype=dtype, device=device))
                nn.init.kaiming_uniform_(w_q_b, a=5**0.5)
                w_q_a_ln = nn.Parameter(torch.empty(0, dtype=dtype, device=device))

            w_kv_a_ln = nn.Parameter(torch.ones(kv_lora_rank, dtype=dtype, device=device))
            self._cache[cache_key] = (w_qkv_a, w_q_b, w_q_a_ln, w_kv_a_ln)
            torch.set_rng_state(rng_state)

        w_qkv_a, w_q_b, w_q_a_ln, w_kv_a_ln = self._cache[cache_key]

        qkv_lora = F.linear(hidden_states, w_qkv_a)
        if q_lora_rank is not None:
            q_c = qkv_lora[:, :q_lora_rank].contiguous()
            kv_lora = qkv_lora[:, q_lora_rank:].contiguous()
            q_c = F.rms_norm(q_c.float(), (q_lora_rank,),
                             weight=w_q_a_ln).to(dtype)
            q = F.linear(q_c, w_q_b)
            q = q.view(num_tokens, num_heads, qk_head_dim)
        else:
            kv_lora = qkv_lora[:, :kv_lora_rank + qk_rope_head_dim].contiguous()
            q = F.linear(hidden_states, w_q_b)
            q = q.view(num_tokens, num_heads, qk_head_dim)

        kv_c = kv_lora[:, :kv_lora_rank].contiguous()
        k_pe = kv_lora[:, kv_lora_rank:kv_lora_rank +
                       qk_rope_head_dim].contiguous()
        kv_c_normed = F.rms_norm(kv_c.float(), (kv_lora_rank,),
                                 weight=w_kv_a_ln).to(dtype)
        k_pe = k_pe.unsqueeze(1)

        # RoPE
        positions = positions.to(device)
        half_rope = qk_rope_head_dim // 2
        arange_tensor = torch.arange(0, qk_rope_head_dim, 2, device=device,
                                     dtype=torch.float32)
        inv_freq = 1.0 / (rope_base **
                          (arange_tensor / qk_rope_head_dim))
        freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos = freqs.cos()
        sin = freqs.sin()

        k_pe_2d = k_pe.squeeze(1)
        k1 = k_pe_2d[..., 0::2]
        k2 = k_pe_2d[..., 1::2]
        k_rot = k1 * cos - k2 * sin
        k_img = k1 * sin + k2 * cos
        k_pe_rotated = torch.stack([k_rot, k_img], dim=-1).flatten(
            -2).to(dtype).unsqueeze(1)

        q_nope = q[..., :qk_nope_head_dim]
        q_rope = q[..., qk_nope_head_dim:]
        q1 = q_rope[..., 0::2]
        q2 = q_rope[..., 1::2]
        q_rot = q1 * cos.unsqueeze(1) - q2 * sin.unsqueeze(1)
        q_img = q1 * sin.unsqueeze(1) + q2 * cos.unsqueeze(1)
        q_rotated = torch.stack([q_rot, q_img], dim=-1).flatten(-2).to(dtype)

        q_combined = torch.cat([q_nope, q_rotated], dim=-1)

        return q_combined, kv_c_normed, k_pe_rotated


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__),"102_MlaPreprocessOperation.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype_):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype_ is torch.bfloat16:
                return torch.normal(mu, sigma, shape,
                                    dtype=torch.float32).to(dtype_)
            return torch.normal(mu, sigma, shape, dtype=dtype_)
        else:
            return torch.empty(shape, dtype=dtype_).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        hs_info = next(inp for inp in inputs if inp["name"] == "hidden_states")
        pos_info = next(inp for inp in inputs if inp["name"] == "positions")
        dtype = dtype_map[hs_info["dtype"]]
        hidden_states = random_tensor(hs_info["shape"], dtype)
        positions = torch.randint(0, 4096, pos_info["shape"],
                                  dtype=torch.int64)
        kv_lora_rank = next(inp for inp in inputs if
                            inp["name"] == "kv_lora_rank")["value"]
        qk_rope_head_dim = next(inp for inp in inputs if
                                inp["name"] == "qk_rope_head_dim")["value"]
        num_heads = next(inp for inp in inputs if
                         inp["name"] == "num_heads")["value"]
        qk_nope_head_dim = next(inp for inp in inputs if
                                inp["name"] == "qk_nope_head_dim")["value"]
        rope_base = next(inp for inp in inputs if
                         inp["name"] == "rope_base")["value"]
        q_lora_rank = next((inp["value"] for inp in inputs if
                            inp["name"] == "q_lora_rank"), None)
        input_groups.append([hidden_states, positions, kv_lora_rank,
                             qk_rope_head_dim, num_heads, qk_nope_head_dim,
                             rope_base, q_lora_rank])
    return input_groups


def get_init_inputs():
    return []
