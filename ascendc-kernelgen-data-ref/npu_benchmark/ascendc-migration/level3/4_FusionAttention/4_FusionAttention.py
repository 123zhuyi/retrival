import torch
import torch_npu
import json
import os


class Model(torch.nn.Module):
    r"""
    torch_npu.npu_fusion_attention = npu_fusion_attention(query, key, value, head_num, input_layout, pse=None, padding_mask=None, atten_mask=None, scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout="", sink=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)

    Fused Transformer Attention Score computation. Supports BSH, SBH, BNSD, BSND, TND layouts, sparse modes, dropout, position encoding, variable-length sequences.

    Returns: (attention_out, softmax_max, softmax_sum, reserved, seed, offset, mask_length).

    Args:
        query (Tensor): Query tensor. Supports fp16, bf16, fp32.
        key (Tensor): Key tensor, same dtype as query.
        value (Tensor): Value tensor, same dtype as query.
        head_num (int): Number of attention heads.
        input_layout (str): Tensor layout, one of "BSH", "SBH", "BNSD", "BSND", "TND".
        pse (Tensor, optional): Position encoding tensor.
        padding_mask (Tensor, optional): Padding mask.
        atten_mask (Tensor, optional): Attention mask. bool or uint8.
        scale (float, optional): Scale factor for QK^T. Default: 1.0.
        keep_prob (float, optional): Dropout keep probability. Default: 1.0.
        pre_tockens (int, optional): Number of tokens to attend to before current position. Default: 2147483647.
        next_tockens (int, optional): Number of tokens to attend to after current position. Default: 2147483647.
        inner_precise (int, optional): Computation precision mode. 0=high_precision, 1=high_performance, 2=high_performance_l1. Default: 0.
        prefix (Tensor, optional): Prefix tensor.
        actual_seq_qlen (Tensor, optional): Actual query sequence lengths for TND layout.
        actual_seq_kvlen (Tensor, optional): Actual key/value sequence lengths for TND layout.
        sparse_mode (int, optional): Sparse attention mode. Default: 0.
        gen_mask_parallel (bool, optional): Generate mask in parallel. Default: True.
        sync (bool, optional): Synchronous execution. Default: False.
        softmax_layout (str, optional): Softmax layout. Default: "".
        sink (Tensor, optional): Sink token tensor.

    Example::

        >>> query = torch.randn(1, 128, 1024, dtype=torch.float16).npu()
        >>> key = torch.randn(1, 128, 1024, dtype=torch.float16).npu()
        >>> value = torch.randn(1, 128, 1024, dtype=torch.float16).npu()
        >>> out = torch_npu.npu_fusion_attention(query, key, value, 8, "BSH")
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                head_num: int, input_layout: str, pse=None, padding_mask=None,
                atten_mask=None, scale=1., keep_prob=1., pre_tockens=2147483647,
                next_tockens=2147483647, inner_precise=0, prefix=None,
                actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num, input_layout,
            pse=pse, padding_mask=padding_mask, atten_mask=atten_mask,
            scale=scale, keep_prob=keep_prob, pre_tockens=pre_tockens,
            next_tockens=next_tockens, inner_precise=inner_precise,
            prefix=prefix, actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen, sparse_mode=sparse_mode,
            gen_mask_parallel=gen_mask_parallel, sync=sync,
            softmax_layout=softmax_layout, sink=sink,
        )


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "4_FusionAttention.json")

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

        # Build lookup by name
        input_by_name = {inp["name"]: inp for inp in inputs}

        # Required tensor inputs
        query_info = input_by_name["query"]
        key_info = input_by_name["key"]
        value_info = input_by_name["value"]

        dtype_str = query_info["dtype"]
        dtype = dtype_map.get(dtype_str, torch.float16)

        # Helper to create tensor
        def _make_tensor(info, idx, dtype_override=None):
            dt = dtype_override if dtype_override else dtype_map.get(info.get("dtype", dtype_str), torch.float16)
            shape = info["shape"]
            fill_val = info.get("fill", None)
            if fill_val in fill_map:
                return torch.full(shape, fill_map[fill_val], dtype=dt)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, shape, dtype=torch.float32)
                return raw.to(dt)
            else:
                raw = torch.empty(shape, dtype=torch.float32).uniform_(-5, 5)
                return raw.to(dt)

        query = _make_tensor(query_info, idx)
        key = _make_tensor(key_info, idx, dtype_override=dtype_map.get(key_info.get("dtype", dtype_str), torch.float16))
        value = _make_tensor(value_info, idx, dtype_override=dtype_map.get(value_info.get("dtype", dtype_str), torch.float16))

        head_num = input_by_name["head_num"]["value"]
        input_layout = input_by_name["input_layout"]["value"]
        scale = input_by_name.get("scale", {}).get("value", 1.0)
        keep_prob = input_by_name.get("keep_prob", {}).get("value", 1.0)
        pre_tockens = input_by_name.get("pre_tockens", {}).get("value", 2147483647)
        next_tockens = input_by_name.get("next_tockens", {}).get("value", 2147483647)
        inner_precise = input_by_name.get("inner_precise", {}).get("value", 0)
        sparse_mode = input_by_name.get("sparse_mode", {}).get("value", 0)
        gen_mask_parallel = input_by_name.get("gen_mask_parallel", {}).get("value", True)
        sync = input_by_name.get("sync", {}).get("value", False)

        # Optional tensors: pse, atten_mask, sink, padding_mask
        pse = None
        if "pse" in input_by_name:
            pse_info = input_by_name["pse"]
            pse_dtype = dtype_map.get(pse_info.get("dtype", "float16"), torch.float16)
            pse = _make_tensor(pse_info, idx, dtype_override=pse_dtype)

        atten_mask = None
        if "atten_mask" in input_by_name:
            atten_info = input_by_name["atten_mask"]
            atten_dtype_str = atten_info.get("dtype", "bool")
            if atten_dtype_str == "bool":
                atten_dtype = torch.bool
            elif atten_dtype_str == "uint8":
                atten_dtype = torch.uint8
            else:
                atten_dtype = torch.bool
            atten_shape = atten_info["shape"]
            atten_fill = atten_info.get("fill", None)
            if atten_fill in fill_map:
                atten_mask = torch.full(atten_shape, fill_map[atten_fill], dtype=atten_dtype)
            elif idx % 2 == 0:
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                raw = torch.normal(mu, sigma, atten_shape, dtype=torch.float32)
                atten_mask = raw.to(atten_dtype)
            else:
                raw = torch.empty(atten_shape, dtype=torch.float32).uniform_(-5, 5)
                atten_mask = raw.to(atten_dtype)

        sink = None
        if "sink" in input_by_name:
            sink_info = input_by_name["sink"]
            sink_dtype = dtype_map.get(sink_info.get("dtype", "float32"), torch.float32)
            sink = _make_tensor(sink_info, idx, dtype_override=sink_dtype)

        input_groups.append([
            query, key, value, head_num, input_layout,
            pse, None, atten_mask, scale, keep_prob,
            pre_tockens, next_tockens, inner_precise,
            None, None, None, sparse_mode, gen_mask_parallel,
            sync, "", sink,
        ])

    return input_groups


def get_init_inputs():
    return []
