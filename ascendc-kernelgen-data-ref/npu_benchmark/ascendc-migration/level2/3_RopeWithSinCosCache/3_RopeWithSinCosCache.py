import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    RopeWithSinCosCache (RoPE with precomputed Sin/Cos cache).

    Applies Rotary Position Embedding (RoPE) to query and key tensors using
    precomputed sin/cos cache. Supports multi-modal RoPE (mRope) variants
    including Qwen2-VL, Qwen3-VL and Hunyuan modes.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def rope_with_sin_cos_cache_native(positions, query, key, cos_sin_cache,
                                           mrope_section, head_size, num_q_heads,
                                           num_kv_heads, is_neox_style=True, cache_mode=0):
            # 1. Select cos/sin from cache by positions
            # 2. Split into cos and sin halves
            # 3. Apply rotary transform: [x1*cos - x2*sin, x2*cos + x1*sin]
            # 4. Handle mrope_section for multi-modal split
            # 5. Return rotated query and key

    Args:
        positions (Tensor): Position indices, [num_tokens] or [3, num_tokens].
        query (Tensor): Query tensor, [num_tokens, num_q_heads * head_size].
        key (Tensor): Key tensor, [num_tokens, num_kv_heads * head_size].
        cos_sin_cache (Tensor): Precomputed cos/sin cache.
        mrope_section (list): Section lengths for multi-modal RoPE.
        head_size (int): Dimension per head.
        is_neox_style (bool, optional): Use interleaved (GPT-NeoX) style. Default: True.
        cache_mode (int, optional): 0=Qwen2-VL, 1=Qwen3-VL, 4-dim section=Hunyuan. Default: 0.

    Returns:
        tuple: (rotated_query, rotated_key)

    Example::

        >>> positions = torch.arange(4)
        >>> query = torch.randn(4, 32)
        >>> key = torch.randn(4, 16)
        >>> cos_sin_cache = torch.randn(128, 16)
        >>> mrope_section = [8, 8]
        >>> q, k = rope_with_sin_cos_cache_native(positions, query, key, cos_sin_cache,
        ...                                       mrope_section, 16, 2, 1)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor,
                cos_sin_cache: torch.Tensor, mrope_section: list, head_size: int,
                is_neox_style: bool = True, cache_mode: int = 0) -> tuple:
        """
        Applies RoPE with sin/cos cache to query and key.

        Args:
            positions (torch.Tensor): Position indices.
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            cos_sin_cache (torch.Tensor): Precomputed cos/sin cache.
            mrope_section (list): Section lengths for multi-modal RoPE.
            head_size (int): Dimension per head.
            is_neox_style (bool, optional): Use interleaved style. Default: True.
            cache_mode (int, optional): Cache mode. Default: 0.

        Returns:
            tuple: (rotated_query, rotated_key).
        """
        num_tokens = positions.shape[-1]
        rotary_dim = cos_sin_cache.shape[-1]
        num_q_heads = query.shape[1] // head_size
        num_kv_heads = key.shape[1] // head_size

        qwen2_vl = (cache_mode != 1 and len(mrope_section) != 4)
        qwen3_vl = (cache_mode == 1)
        hunyuan = (len(mrope_section) == 4)

        if qwen2_vl:
            positions_flatten = positions.flatten()
            cos_sin = cos_sin_cache.index_select(0, positions_flatten)
            cos_sin = cos_sin.reshape(-1, rotary_dim)
        elif qwen3_vl or hunyuan:
            cos_sin = cos_sin_cache[positions]

        cos, sin = cos_sin.chunk(2, dim=-1)

        if positions.ndim == 2:
            if qwen2_vl:
                cos = cos.reshape(3, -1, rotary_dim // 2)
                cos_0 = cos[0, :, :mrope_section[0]]
                cos_1 = cos[1, :, mrope_section[0]:(mrope_section[0] + mrope_section[1])]
                cos_2 = cos[2, :, (mrope_section[0] + mrope_section[1]):(
                    mrope_section[0] + mrope_section[1] + mrope_section[2])]
                cos = torch.concat((cos_0, cos_1, cos_2), dim=-1)

                sin = sin.reshape(3, -1, rotary_dim // 2)
                sin_0 = sin[0, :, :mrope_section[0]]
                sin_1 = sin[1, :, mrope_section[0]:(mrope_section[0] + mrope_section[1])]
                sin_2 = sin[2, :, (mrope_section[0] + mrope_section[1]):(
                    mrope_section[0] + mrope_section[1] + mrope_section[2])]
                sin = torch.concat((sin_0, sin_1, sin_2), dim=-1)
            elif qwen3_vl:
                cos_t = cos[0].clone()
                cos_t[..., 1:mrope_section[1] * 3:3] = cos[1, ..., 1:mrope_section[1] * 3:3]
                cos_t[..., 2:mrope_section[1] * 3:3] = cos[2, ..., 2:mrope_section[1] * 3:3]
                cos = cos_t

                sin_t = sin[0].clone()
                sin_t[..., 1:mrope_section[1] * 3:3] = sin[1, ..., 1:mrope_section[1] * 3:3]
                sin_t[..., 2:mrope_section[1] * 3:3] = sin[2, ..., 2:mrope_section[1] * 3:3]
                sin = sin_t
            elif hunyuan:
                cos = torch.cat(
                    [m[i] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
                )
                sin = torch.cat(
                    [m[i] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
                )

        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        if is_neox_style:
            x1, x2 = torch.chunk(query_rot, 2, dim=-1)
        else:
            x1 = query_rot[..., ::2]
            x2 = query_rot[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            query_rot = torch.cat((o1, o2), dim=-1)
        else:
            query_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]
        if is_neox_style:
            x1, x2 = torch.chunk(key_rot, 2, dim=-1)
        else:
            x1 = key_rot[..., ::2]
            x2 = key_rot[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            key_rot = torch.cat((o1, o2), dim=-1)
        else:
            key_rot = torch.stack((o1, o2), dim=-1).flatten(-2)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        return query, key


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "3_RopeWithSinCosCache.json")

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

        positions_info = inputs[0]
        query_info = inputs[1]
        key_info = inputs[2]
        cos_sin_cache_info = inputs[3]
        mrope_section_info = inputs[4]
        head_size_info = inputs[5]
        is_neox_style_info = inputs[6] if len(inputs) > 6 else {"value": True}
        cache_mode_info = inputs[7] if len(inputs) > 7 else {"value": 0}

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

        positions = make_tensor(positions_info, idx)
        query = make_tensor(query_info, idx)
        key = make_tensor(key_info, idx)
        cos_sin_cache = make_tensor(cos_sin_cache_info, idx)

        mrope_section = mrope_section_info["value"]
        head_size = head_size_info["value"]
        is_neox_style = is_neox_style_info["value"]
        cache_mode = cache_mode_info["value"]

        input_groups.append([
            positions, query, key, cos_sin_cache,
            mrope_section, head_size, is_neox_style, cache_mode
        ])

    return input_groups


def get_init_inputs():
    return []