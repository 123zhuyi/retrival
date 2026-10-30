import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs fused inference attention score computation using NPU accelerated npu_fused_infer_attention_score.
    Supports both prompt (full) and incremental inference scenarios for FlashAttention.

    Torch native implementation (equivalent to npu_fused_infer_attention_score for standard MHA path):
        # 1. Infer shapes and reshape to BNSD based on input_layout
        if input_layout == "BSH":
            batch_size, seq_len, hidden = query.shape
            _, seq_len_kv, _ = key.shape
            head_dim = hidden // num_heads
            q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = key.view(batch_size, seq_len_kv, num_key_value_heads or num_heads, head_dim).transpose(1, 2)
            v = value.view(batch_size, seq_len_kv, num_key_value_heads or num_heads, head_dim).transpose(1, 2)
        elif input_layout == "BNSD":
            q, k, v = query, key, value
            batch_size = q.shape[0]
            seq_len = q.shape[2]
            seq_len_kv = k.shape[2]
            head_dim = q.shape[-1]
        else:
            raise NotImplementedError(f"Layout {input_layout} not supported in native impl")

        # 2. GQA: repeat k/v heads to match q heads
        if num_key_value_heads and num_key_value_heads != num_heads:
            n_rep = num_heads // num_key_value_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # 3. Compute attention scores: Q @ K^T * scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 4. Apply attention mask (bool mask or additive mask)
        if atten_mask is not None:
            if atten_mask.dtype == torch.bool:
                scores = scores.masked_fill(atten_mask, float('-inf'))
            else:
                scores = scores + atten_mask

        # 5. Softmax over the last dimension
        attn_weights = torch.softmax(scores, dim=-1)

        # 6. Apply attention weights to value
        out = torch.matmul(attn_weights, v)

        # 7. Reshape back to BSH
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return out, None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                pse_shift=None, atten_mask=None, actual_seq_lengths=None,
                actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None,
                dequant_scale2=None, quant_scale2=None, quant_offset2=None,
                antiquant_scale=None, antiquant_offset=None, block_table=None,
                query_padding_size=None, kv_padding_size=None,
                key_antiquant_scale=None, key_antiquant_offset=None,
                value_antiquant_scale=None, value_antiquant_offset=None,
                key_shared_prefix=None, value_shared_prefix=None,
                actual_shared_prefix_len=None, query_rope=None, key_rope=None,
                key_rope_antiquant_scale=None, num_heads=1, scale=1.0,
                pre_tokens=2147483647, next_tokens=2147483647,
                input_layout="BSH", num_key_value_heads=0, sparse_mode=0,
                inner_precise=0, block_size=0, antiquant_mode=0,
                softmax_lse_flag=False, key_antiquant_mode=0, value_antiquant_mode=0):
        """
        Performs fused inference attention score computation on NPU.

        Args:
            query (Tensor): Query input, dtype float16/bfloat16.
            key (Tensor): Key input, dtype float16/bfloat16/int8/int4.
            value (Tensor): Value input, dtype float16/bfloat16/int8/int4.
            pse_shift (Tensor, optional): Position encoding parameter.
            atten_mask (Tensor, optional): Attention mask, dtype bool/int8/uint8.
            actual_seq_lengths (List[int], optional): Valid seqlen per batch for query.
            actual_seq_lengths_kv (List[int], optional): Valid seqlen per batch for key/value.
            dequant_scale1 (Tensor, optional): BMM1 dequantization factor, dtype uint64/float32.
            quant_scale1 (Tensor, optional): BMM2 quantization factor, dtype float32.
            dequant_scale2 (Tensor, optional): BMM2 dequantization factor, dtype uint64/float32.
            quant_scale2 (Tensor, optional): Output quantization factor, dtype float32/bfloat16.
            quant_offset2 (Tensor, optional): Output quantization offset, dtype float32/bfloat16.
            antiquant_scale (Tensor, optional): Anti-quantization scale, dtype float16/bfloat16.
            antiquant_offset (Tensor, optional): Anti-quantization offset, dtype float16/bfloat16.
            block_table (Tensor, optional): PageAttention block mapping table, dtype int32.
            query_padding_size (Tensor, optional): Query right-alignment info, dtype int64.
            kv_padding_size (Tensor, optional): Key/value right-alignment info, dtype int64.
            key_antiquant_scale (Tensor, optional): Key anti-quantization scale.
            key_antiquant_offset (Tensor, optional): Key anti-quantization offset.
            value_antiquant_scale (Tensor, optional): Value anti-quantization scale.
            value_antiquant_offset (Tensor, optional): Value anti-quantization offset.
            key_shared_prefix (Tensor, optional): Key shared prefix.
            value_shared_prefix (Tensor, optional): Value shared prefix.
            actual_shared_prefix_len (List[int], optional): Shared prefix valid seqlen.
            query_rope (Tensor, optional): Query rope info for MLA.
            key_rope (Tensor, optional): Key rope info for MLA.
            key_rope_antiquant_scale (Tensor, optional): Reserved parameter.
            num_heads (int): Number of query heads, default 1.
            scale (float): Scaling factor, default 1.0.
            pre_tokens (int): Forward token count for sparse, default 2147483647.
            next_tokens (int): Backward token count for sparse, default 2147483647.
            input_layout (str): Data layout, supports BSH/BSND/BNSD/TND etc., default "BSH".
            num_key_value_heads (int): Number of key/value heads for GQA, default 0.
                Note: 0 means MHA (non-GQA), i.e., num_key_value_heads equals num_heads.
            sparse_mode (int): Sparse mode, default 0.
            inner_precise (int): Precision/performance mode, default 0.
            block_size (int): PageAttention block size, default 0.
            antiquant_mode (int): Anti-quantization mode, 0=perchannel, 1=pertoken, default 0.
            softmax_lse_flag (bool): Whether to output softmax_lse, default False.
            key_antiquant_mode (int): Key anti-quantization mode, default 0.
            value_antiquant_mode (int): Value anti-quantization mode, default 0.

        Returns:
            tuple: (attention_out, softmax_lse)
        """
        import torch_npu

        # Defensive fix: num_key_value_heads=0 indicates MHA (non-GQA) where
        # num_key_value_heads should equal num_heads. Some NPU kernels may
        # misinterpret 0 as 1, causing head_dim mismatch.
        effective_nkv = num_key_value_heads if num_key_value_heads > 0 else num_heads

        return torch_npu.npu_fused_infer_attention_score(
            query, key, value,
            pse_shift=pse_shift, atten_mask=atten_mask,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            dequant_scale1=dequant_scale1, quant_scale1=quant_scale1,
            dequant_scale2=dequant_scale2, quant_scale2=quant_scale2,
            quant_offset2=quant_offset2,
            antiquant_scale=antiquant_scale, antiquant_offset=antiquant_offset,
            block_table=block_table,
            query_padding_size=query_padding_size, kv_padding_size=kv_padding_size,
            key_antiquant_scale=key_antiquant_scale, key_antiquant_offset=key_antiquant_offset,
            value_antiquant_scale=value_antiquant_scale, value_antiquant_offset=value_antiquant_offset,
            key_shared_prefix=key_shared_prefix, value_shared_prefix=value_shared_prefix,
            actual_shared_prefix_len=actual_shared_prefix_len,
            query_rope=query_rope, key_rope=key_rope,
            key_rope_antiquant_scale=key_rope_antiquant_scale,
            num_heads=num_heads, scale=scale,
            pre_tokens=pre_tokens, next_tokens=next_tokens,
            input_layout=input_layout, num_key_value_heads=effective_nkv,
            sparse_mode=sparse_mode, inner_precise=inner_precise,
            block_size=block_size, antiquant_mode=antiquant_mode,
            softmax_lse_flag=softmax_lse_flag,
            key_antiquant_mode=key_antiquant_mode,
            value_antiquant_mode=value_antiquant_mode)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "7_FusedInferAttentionScore.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        """独立随机选择正态/均匀分布"""
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
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
            "int8": torch.int8,
            "int32": torch.int32,
        }

        query_info = inputs[0]
        key_info = inputs[1]
        value_info = inputs[2]

        dtype = dtype_map[query_info["dtype"]]
        key_dtype = dtype_map.get(key_info.get("dtype", query_info["dtype"]), dtype)

        # 浮点 tensor 改用独立随机分布
        query = random_tensor(query_info["shape"], dtype)
        key = random_tensor(key_info["shape"], key_dtype)
        value = random_tensor(value_info["shape"], key_dtype)

        pse_shift = None
        atten_mask = None
        actual_seq_lengths = None
        actual_seq_lengths_kv = None
        dequant_scale1 = None
        quant_scale1 = None
        dequant_scale2 = None
        quant_scale2 = None
        quant_offset2 = None
        antiquant_scale = None
        antiquant_offset = None
        block_table = None
        query_padding_size = None
        kv_padding_size = None
        key_antiquant_scale = None
        key_antiquant_offset = None
        value_antiquant_scale = None
        value_antiquant_offset = None
        key_shared_prefix = None
        value_shared_prefix = None
        actual_shared_prefix_len = None
        query_rope = None
        key_rope = None
        key_rope_antiquant_scale = None
        num_heads = 1
        scale = 1.0
        pre_tokens = 2147483647
        next_tokens = 2147483647
        input_layout = "BSH"
        num_key_value_heads = 0
        sparse_mode = 0
        inner_precise = 0
        block_size = 0
        antiquant_mode = 0
        softmax_lse_flag = False
        key_antiquant_mode = 0
        value_antiquant_mode = 0

        for inp in inputs[3:]:
            name = inp.get("name", "")
            if name == "pse_shift":
                pse_shift = random_tensor(inp["shape"], dtype)
            elif name == "atten_mask":
                atten_mask_dtype_map = {
                    "bool": torch.bool,
                    "int8": torch.int8,
                    "uint8": torch.uint8,
                }
                atten_mask_dtype = atten_mask_dtype_map.get(inp.get("dtype", "bool"), torch.bool)
                atten_mask = torch.ones(inp["shape"], dtype=atten_mask_dtype)
            elif name == "actual_seq_lengths":
                actual_seq_lengths = inp["value"]
            elif name == "actual_seq_lengths_kv":
                actual_seq_lengths_kv = inp["value"]
            elif name == "dequant_scale1":
                dequant_scale1 = random_tensor(inp["shape"], torch.float32)
            elif name == "quant_scale1":
                quant_scale1 = random_tensor(inp["shape"], torch.float32)
            elif name == "dequant_scale2":
                dequant_scale2 = random_tensor(inp["shape"], torch.float32)
            elif name == "quant_scale2":
                quant_scale2 = random_tensor(inp["shape"], torch.float32)
            elif name == "quant_offset2":
                quant_offset2 = random_tensor(inp["shape"], torch.float32)
            elif name == "antiquant_scale":
                antiquant_scale = random_tensor(inp["shape"], dtype)
            elif name == "antiquant_offset":
                antiquant_offset = random_tensor(inp["shape"], dtype)
            elif name == "block_table":
                block_table = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "query_padding_size":
                query_padding_size = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "kv_padding_size":
                kv_padding_size = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "key_antiquant_scale":
                key_antiquant_scale = random_tensor(inp["shape"], dtype)
            elif name == "key_antiquant_offset":
                key_antiquant_offset = random_tensor(inp["shape"], dtype)
            elif name == "value_antiquant_scale":
                value_antiquant_scale = random_tensor(inp["shape"], dtype)
            elif name == "value_antiquant_offset":
                value_antiquant_offset = random_tensor(inp["shape"], dtype)
            elif name == "key_shared_prefix":
                key_shared_prefix = random_tensor(inp["shape"], key_dtype)
            elif name == "value_shared_prefix":
                value_shared_prefix = random_tensor(inp["shape"], key_dtype)
            elif name == "actual_shared_prefix_len":
                actual_shared_prefix_len = inp["value"]
            elif name == "query_rope":
                query_rope = random_tensor(inp["shape"], dtype)
            elif name == "key_rope":
                key_rope = random_tensor(inp["shape"], dtype)
            elif name == "key_rope_antiquant_scale":
                key_rope_antiquant_scale = random_tensor(inp["shape"], dtype)
            elif name == "num_heads":
                num_heads = inp["value"]
            elif name == "scale":
                scale = inp["value"]
            elif name == "pre_tokens":
                pre_tokens = inp["value"]
            elif name == "next_tokens":
                next_tokens = inp["value"]
            elif name == "input_layout":
                input_layout = inp["value"]
            elif name == "num_key_value_heads":
                num_key_value_heads = inp["value"]
            elif name == "sparse_mode":
                sparse_mode = inp["value"]
            elif name == "inner_precise":
                inner_precise = inp["value"]
            elif name == "block_size":
                block_size = inp["value"]
            elif name == "antiquant_mode":
                antiquant_mode = inp["value"]
            elif name == "softmax_lse_flag":
                softmax_lse_flag = inp["value"]
            elif name == "key_antiquant_mode":
                key_antiquant_mode = inp["value"]
            elif name == "value_antiquant_mode":
                value_antiquant_mode = inp["value"]

        # ========== 新增：BSH layout GQA case 的 key/value shape 防御性调整 ==========
        # NPU 算子要求 query_head_dim >= value_head_dim。
        # BSH layout 下 head_dim = hidden_size // num_heads。
        # 当 nkv < num_heads 时，若 key/value hidden_size 与 query 相同，则 value_head_dim > query_head_dim，导致报错。
        if input_layout == "BSH" and 0 < num_key_value_heads < num_heads:
            if query.shape[-1] % num_heads == 0:
                head_dim = query.shape[-1] // num_heads
                expected_kv_hidden = num_key_value_heads * head_dim
                if key.shape[-1] != expected_kv_hidden:
                    key = random_tensor(list(key.shape[:-1]) + [expected_kv_hidden], key_dtype)
                if value.shape[-1] != expected_kv_hidden:
                    value = random_tensor(list(value.shape[:-1]) + [expected_kv_hidden], key_dtype)

        input_groups.append([query, key, value,
                             pse_shift, atten_mask, actual_seq_lengths,
                             actual_seq_lengths_kv, dequant_scale1, quant_scale1,
                             dequant_scale2, quant_scale2, quant_offset2,
                             antiquant_scale, antiquant_offset, block_table,
                             query_padding_size, kv_padding_size,
                             key_antiquant_scale, key_antiquant_offset,
                             value_antiquant_scale, value_antiquant_offset,
                             key_shared_prefix, value_shared_prefix,
                             actual_shared_prefix_len, query_rope, key_rope,
                             key_rope_antiquant_scale, num_heads, scale,
                             pre_tokens, next_tokens, input_layout,
                             num_key_value_heads, sparse_mode, inner_precise,
                             block_size, antiquant_mode, softmax_lse_flag,
                             key_antiquant_mode, value_antiquant_mode])
    return input_groups


def get_init_inputs():
    return []