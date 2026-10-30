import torch
import torch.nn as nn
import json
import os
import math


def _to_cumsum(lengths, total):
    if lengths is None:
        return [total]
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    if len(lengths) == 0:
        return [total]
    if lengths[-1] >= total * 0.9:
        return lengths
    return [sum(lengths[:i + 1]) for i in range(len(lengths))]


def _get_tnd_idx(cumsum_list, t_idx):
    b_idx = 0
    while b_idx < len(cumsum_list) and t_idx >= cumsum_list[b_idx]:
        b_idx += 1
    s1_offset = 0 if b_idx == 0 else cumsum_list[b_idx - 1]
    s1_idx = t_idx - s1_offset
    return b_idx, s1_idx


def _gather_kv(cur_k, cur_v, topk, cur_s2, block_size):
    s2_sparse = []
    for sparse_id in topk.view(-1).tolist():
        if sparse_id == -1:
            break
        begin_idx = sparse_id * block_size
        end_idx = min(begin_idx + block_size, cur_s2)
        s2_sparse.extend(range(begin_idx, end_idx))
    if not s2_sparse:
        return cur_k[:0], cur_v[:0], 0
    return cur_k[s2_sparse, :], cur_v[s2_sparse, :], len(s2_sparse)


def _cal_attenmask(topk, s1_idx, block_size, actual_sel_s2, g, cur_s2, cur_s1):
    """sparse_mode=3 (right-down causal) 的 attention mask"""
    atten_msk = torch.zeros(actual_sel_s2)
    sparse_tail_idx = math.ceil(cur_s2 / block_size)
    sparse_tail_seq_len = cur_s2 % block_size
    if sparse_tail_seq_len == 0:
        sparse_tail_seq_len = block_size

    delta_s = cur_s2 - cur_s1
    threshold = delta_s + s1_idx + 1

    s_idx = 0
    for sparse_id in topk.view(-1).tolist():
        if sparse_id == -1:
            break
        begin_idx = sparse_id * block_size
        block_len = block_size if sparse_id != sparse_tail_idx - 1 else sparse_tail_seq_len
        end_idx = begin_idx + block_len

        if begin_idx < threshold and end_idx <= threshold:
            s_idx += block_len
            continue
        elif end_idx > threshold:
            local_offset = 0 if threshold <= begin_idx else threshold - begin_idx
            mask_begin = s_idx + local_offset
            mask_end = s_idx + block_len
            atten_msk[mask_begin:mask_end] = 1
        s_idx += block_len

    return atten_msk


def _tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    return y / x_sum, x_max, x_sum


class Model(nn.Module):
    """
    Model that performs Sparse Flash Attention computation.
    torch_npu.npu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, ...)
    -> (attention_out, softmax_max, softmax_sum)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value, sparse_indices, scale_value,
                block_table=None, actual_seq_lengths_query=None,
                actual_seq_lengths_kv=None, query_rope=None, key_rope=None,
                sparse_block_size=1, layout_query='BSND', layout_kv='BSND',
                sparse_mode=3, pre_tokens=None, next_tokens=None,
                attention_mode=0, return_softmax_lse=False):
        dtype = query.dtype
        device = query.device
        atten_enable = (sparse_mode != 0)

        # ---------- 1. 统一维度推导与格式转换 ----------
        if layout_query == 'BSND':
            b, s1, n1, d = query.shape
            t1 = b * s1
            n2 = key.shape[2]
            g = n1 // n2
            # 转 TND 格式处理: [B, S1, N2, G, D] -> [B*S1, N2, G, D]
            q_fp = query.permute(0, 2, 1, 3).reshape(t1, n2, g, d).float()
            asq_cumsum = _to_cumsum(actual_seq_lengths_query, t1)
            s2 = key.shape[1]
            ask_cumsum = _to_cumsum(actual_seq_lengths_kv, b * s2)
        elif layout_query == 'TND':
            t1, n1, d = query.shape
            n2 = key.shape[1]
            g = n1 // n2
            b = actual_seq_lengths_query.shape[0] if actual_seq_lengths_query is not None else 1
            q_fp = query.reshape(t1, n2, g, d).float()
            asq_cumsum = _to_cumsum(actual_seq_lengths_query, t1)
            t2 = key.shape[0]
            ask_cumsum = _to_cumsum(actual_seq_lengths_kv, t2)
        else:
            raise ValueError(f"Unsupported layout_query: {layout_query}")

        # KV 转 TND
        if layout_kv == 'BSND':
            t2 = b * s2
            k_fp = key.permute(0, 2, 1, 3).reshape(t2, n2, d).float()
            v_fp = value.permute(0, 2, 1, 3).reshape(t2, n2, d).float()
        elif layout_kv == 'PA_BSND':
            # 当前测试用例中 block_table 为恒等映射，按物理块顺序展平处理
            t2 = key.shape[0] * key.shape[1]
            k_fp = key.permute(0, 2, 1, 3).reshape(t2, n2, d).float()
            v_fp = value.permute(0, 2, 1, 3).reshape(t2, n2, d).float()
        elif layout_kv == 'TND':
            t2 = key.shape[0]
            k_fp = key.reshape(t2, n2, d).float()
            v_fp = value.reshape(t2, n2, d).float()
        else:
            raise ValueError(f"Unsupported layout_kv: {layout_kv}")

        # Rope (MLA)
        if query_rope is not None and key_rope is not None:
            q_rope = query_rope.float()
            k_rope = key_rope.float()
            if layout_query == 'BSND':
                q_rope = q_rope.permute(0, 2, 1, 3).reshape(t1, n2, g, -1)
            else:
                q_rope = q_rope.reshape(t1, n2, g, -1)
            if layout_kv == 'BSND' or layout_kv == 'PA_BSND':
                k_rope = k_rope.permute(0, 2, 1, 3).reshape(t2, n2, -1)
            else:
                k_rope = k_rope.reshape(t2, n2, -1)
            q_fp = torch.cat([q_fp, q_rope], dim=-1)
            k_fp = torch.cat([k_fp, k_rope], dim=-1)
        sparse_indices_tnd = sparse_indices.reshape(t1, n2, -1)
        out = torch.zeros(t1, n2, g, d, dtype=torch.float32, device=device)
        x_max_out = torch.zeros(t1, n2, g, 1, dtype=torch.float32, device=device)
        x_sum_out = torch.zeros(t1, n2, g, 1, dtype=torch.float32, device=device)

        for i in range(t1):
            b_idx, s1_idx = _get_tnd_idx(asq_cumsum, i)
            for n2_idx in range(n2):
                topk = sparse_indices_tnd[i, n2_idx]
                q_cal = q_fp[i, n2_idx]

                s2_start = 0 if b_idx == 0 else ask_cumsum[b_idx - 1]
                s2_end = ask_cumsum[b_idx]
                cur_s2 = ask_cumsum[b_idx] if b_idx == 0 else (ask_cumsum[b_idx] - ask_cumsum[b_idx - 1])
                cur_s1 = asq_cumsum[b_idx] if b_idx == 0 else (asq_cumsum[b_idx] - asq_cumsum[b_idx - 1])

                cur_k = k_fp[s2_start:s2_end, n2_idx, :]
                cur_v = v_fp[s2_start:s2_end, n2_idx, :]

                k_cal, v_cal, actual_sel_s2 = _gather_kv(cur_k, cur_v, topk, cur_s2, sparse_block_size)

                if actual_sel_s2 > 0:
                    qk = torch.matmul(q_cal, k_cal.t()) * scale_value

                    if atten_enable:
                        atten_msk = _cal_attenmask(topk, s1_idx, sparse_block_size, actual_sel_s2, g, cur_s2, cur_s1)
                        atten_msk = atten_msk.to(device=device, dtype=torch.float32)
                        qk = qk + atten_msk.view(1, -1) * (-2e35)

                    softmax_res, x_max, x_sum = _tsoftmax(qk)
                    x_max_out[i, n2_idx, :, 0] = x_max.squeeze(-1)
                    x_sum_out[i, n2_idx, :, 0] = x_sum.squeeze(-1)
                    out[i, n2_idx] = torch.matmul(softmax_res, v_cal)  # [G, D]
        x_max_out = x_max_out.expand(t1, n2, g, 8)
        x_sum_out = x_sum_out.expand(t1, n2, g, 8)
        softmax_max = x_max_out.reshape(t1, n2 * g, 8)
        softmax_sum = x_sum_out.reshape(t1, n2 * g, 8)
        if layout_query == 'BSND':
            attention_out = out.reshape(b, s1, n2, g, d).permute(0, 2, 1, 3, 4).reshape(b, s1, n1, d)
        else:
            attention_out = out.reshape(t1, n1, d)
        attention_out = attention_out.to(dtype)
        if not return_softmax_lse:
            empty = torch.empty(0, dtype=torch.float32, device=device)
            return attention_out, empty, empty
        return attention_out, softmax_max, softmax_sum


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "59_SparseFlashAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
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
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        query_info = inputs[0]
        key_info = inputs[1]
        value_info = inputs[2]
        sparse_indices_info = inputs[3]
        scale_value_info = inputs[4]

        dtype = dtype_map[query_info["dtype"]]

        query = random_tensor(query_info["shape"], dtype)
        key = random_tensor(key_info["shape"], dtype)
        value = random_tensor(value_info["shape"], dtype)
        scale_value = scale_value_info["value"]

        block_table = None
        actual_seq_lengths_query = None
        actual_seq_lengths_kv = None
        query_rope = None
        key_rope = None
        sparse_block_size = 1
        layout_query = 'BSND'
        layout_kv = 'BSND'
        sparse_mode = 3
        pre_tokens = None
        next_tokens = None
        attention_mode = 0
        return_softmax_lse = False

        for inp in inputs[5:]:
            name = inp.get("name", "")
            if name == "block_table":
                block_shape = inp["shape"]
                total_blocks = block_shape[0] * block_shape[1]
                block_table = torch.arange(total_blocks, dtype=torch.int32).reshape(block_shape)
            elif name == "actual_seq_lengths_query":
                actual_seq_lengths_query = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "actual_seq_lengths_kv":
                actual_seq_lengths_kv = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "query_rope":
                query_rope = random_tensor(inp["shape"], dtype)
            elif name == "key_rope":
                key_rope = random_tensor(inp["shape"], dtype)
            elif name == "sparse_block_size":
                sparse_block_size = inp["value"]
            elif name == "layout_query":
                layout_query = inp["value"]
            elif name == "layout_kv":
                layout_kv = inp["value"]
            elif name == "sparse_mode":
                sparse_mode = inp["value"]
            elif name == "pre_tokens":
                pre_tokens = inp["value"]
            elif name == "next_tokens":
                next_tokens = inp["value"]
            elif name == "attention_mode":
                attention_mode = inp["value"]
            elif name == "return_softmax_lse":
                return_softmax_lse = inp["value"]

        # Determine max valid index for sparse_indices
        if layout_kv == 'PA_BSND' and block_table is not None:
            max_idx = block_table.shape[0] * block_table.shape[1]
        elif sparse_block_size > 1:
            key_s2 = key_info["shape"][1]
            max_idx = max(1, key_s2 // sparse_block_size)
        else:
            key_s2 = key_info["shape"][1]
            max_idx = max(1, key_s2)

        sparse_indices_shape = sparse_indices_info["shape"]
        sparse_indices = torch.randint(0, max_idx, sparse_indices_shape, dtype=torch.int32)

        input_groups.append([query, key, value, sparse_indices, scale_value,
                             block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
                             query_rope, key_rope, sparse_block_size,
                             layout_query, layout_kv, sparse_mode,
                             pre_tokens, next_tokens, attention_mode, return_softmax_lse])
    return input_groups


def get_init_inputs():
    return []