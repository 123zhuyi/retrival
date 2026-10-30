import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs Lightning Indexer computation using NPU accelerated npu_lightning_indexer.
    Computes Top-k positions for each token based on index queries and keys.

    Workarounds for known issues:
      1. npu_lightning_indexer returns uninitialized memory (NaN/Inf) when return_value=False.
         For non-PA_BSND layouts we always pass return_value=True to the NPU kernel and
         post-process (zero-fill or return real values) in Python.
      2. PA_BSND layout does not support return_value=True. For PA_BSND we pass
         return_value=False to the kernel and construct zero-filled sparse_values in Python.
      3. The native algorithm below aligns mask semantics with the actual NPU behavior:
         masked positions are excluded from top-k (treated as -inf) rather than included as zeros.

    ==================== 对齐说明（与 aclnnLightningIndexer CPU golden 逐点对齐） ====================
    本 docstring 内的 PyTorch native 实现已对齐 golden 语义：
      - score = sum_over_heads( w * ReLU(Q @ K^T) )，无 scale；head 按 [N2, G] 分组（G = N1/N2），
        组内求和（GQA，n2=1 时退化为对全部 N1 个头求和）；
      - sparse_mode=3 使用右下因果 mask：第 i 行仅 j <= i + (actual_s2 - actual_s1) 有效，
        masked 位置置 -inf 不参与 topk；sparse_mode=0 不加 mask；其余 sparse_mode 抛异常；
      - topk 结果按分数降序、同分取下标小者优先（stable sort，与 golden 的 lexsort 一致）；
      - 某行有效 kv 数不足 topk 个数时，超出槽位 indices 填 -1、values 填 -inf；
      - BSND 下 actual_s1 之后的无效行 indices 填 -1、values 填 -inf（不再是 0）；
      - pre_tokens / next_tokens 不参与计算（与 golden 一致，causal mask 完全由 actual_seq 推导）。
    与本 bench 实际 forward 的差异（有意保留）：
      - native 输出按"分数降序"排列；实际 forward 对输出做了 index 升序规范化（方案 A），
        两者比对前需先统一排序；
      - return_value=False 时 native 返回 (indices, zeros) 以保持算子签名，golden 此时只返回
        单个 indices 张量；
      - native 的 values 为 fp32 浮点分值；golden 最终按 case 配置 cast 到 bf16/fp16，
        本 bench 实际 forward cast 到 int32，比对 values 前需统一 dtype。
    ==============================================================================================

    torch_npu.npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3, pre_tokens=None, next_tokens=None, return_value=False) -> (Tensor, Tensor)
    PyTorch native implementation of forward function
    def forward(self, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
                actual_seq_lengths_query=None, actual_seq_lengths_key=None,
                block_table=None, layout_query="BSND", layout_key="BSND",
                sparse_count=2048, sparse_mode=3, pre_tokens=None, next_tokens=None,
                return_value=False):
        if sparse_mode not in (0, 3):
            raise ValueError(f"unsupported sparse_mode: {sparse_mode}")
        # pre_tokens/next_tokens 与 golden 一致，不参与计算
        device = query.device

        def _tolist(x):
            return x.tolist() if isinstance(x, torch.Tensor) else list(x)

        # ---- query layout ----
        asq_cum = None
        if layout_query == "BSND":
            b, s1, n1, d = query.shape
            if actual_seq_lengths_query is None:
                asq_list = [s1] * b
            else:
                asq_list = _tolist(actual_seq_lengths_query)
        elif layout_query == "TND":
            t1, n1, d = query.shape
            if actual_seq_lengths_query is None:
                raise ValueError("actual_seq_lengths_query must be provided for TND layout")
            # 累积长度：保留原始累积值用于切片，另算逐 batch 长度
            asq_cum = _tolist(actual_seq_lengths_query)
            b = len(asq_cum)
            s1 = None
            asq_list = [asq_cum[0]] + [asq_cum[i] - asq_cum[i - 1] for i in range(1, b)]
        else:
            raise ValueError(f"Unsupported layout_query: {layout_query}")

        # ---- key layout ----
        ask_cum = None
        if layout_key == "BSND":
            b_k, s2, n2, d_k = key.shape
            if actual_seq_lengths_key is None:
                ask_list = [s2] * b_k
            else:
                ask_list = _tolist(actual_seq_lengths_key)
        elif layout_key == "PA_BSND":
            block_count, block_size, n2, d_k = key.shape
            if actual_seq_lengths_key is None:
                raise ValueError("actual_seq_lengths_key must be provided for PA_BSND layout")
            ask_list = _tolist(actual_seq_lengths_key)
            b_k = len(ask_list)
            s2 = None
        elif layout_key == "TND":
            t2, n2, d_k = key.shape
            if actual_seq_lengths_key is None:
                raise ValueError("actual_seq_lengths_key must be provided for TND layout")
            ask_cum = _tolist(actual_seq_lengths_key)
            b_k = len(ask_cum)
            s2 = None
            ask_list = [ask_cum[0]] + [ask_cum[i] - ask_cum[i - 1] for i in range(1, b_k)]
        else:
            raise ValueError(f"Unsupported layout_key: {layout_key}")

        assert n1 % n2 == 0, f"n1 must be divisible by n2, got {n1} vs {n2}"
        g = n1 // n2
        assert d == d_k, f"d must equal d_k, got {d} vs {d_k}"
        assert b_k == b, f"batch mismatch: query {b} vs key {b_k}"

        # ---- 输出预分配：无效区域 indices=-1 / values=-inf（对齐 golden） ----
        K = sparse_count
        if layout_query == "BSND":
            indices_out = torch.full((b, s1, n2, K), -1, dtype=torch.int32, device=device)
            values_out = torch.full((b, s1, n2, K), float('-inf'), dtype=torch.float32, device=device)
        else:
            indices_out = torch.full((asq_cum[-1], n2, K), -1, dtype=torch.int32, device=device)
            values_out = torch.full((asq_cum[-1], n2, K), float('-inf'), dtype=torch.float32, device=device)

        for batch_idx in range(b):
            actual_s1 = asq_list[batch_idx]
            actual_s2 = ask_list[batch_idx]
            if actual_s1 == 0:
                continue

            # ---- 切 query / weights ----
            if layout_query == "BSND":
                q_batch = query[batch_idx, :actual_s1]      # [s1', N1, D]
                w_batch = weights[batch_idx, :actual_s1]    # [s1', N1]
                row0 = None
            else:  # TND：用"累积"actseq 在 T 维上切片
                start = 0 if batch_idx == 0 else asq_cum[batch_idx - 1]
                end = asq_cum[batch_idx]
                q_batch = query[start:end]
                w_batch = weights[start:end]
                row0 = start

            # ---- 切 key ----
            if layout_key == "BSND":
                k_batch = key[batch_idx, :actual_s2]        # [s2', N2, D]
            elif layout_key == "PA_BSND":
                bt_row = block_table[batch_idx]
                k_batch = key[bt_row].reshape(-1, n2, d_k)[:actual_s2]
            else:  # TND
                start2 = 0 if batch_idx == 0 else ask_cum[batch_idx - 1]
                end2 = ask_cum[batch_idx]
                k_batch = key[start2:end2]

            # ---- 打分：score = sum_heads( w * ReLU(Q @ K^T) )，head 按 [N2, G] 分组 ----
            q_fp32 = q_batch.to(torch.float32)
            k_fp32 = k_batch.to(torch.float32)
            w_fp32 = w_batch.to(torch.float32)
            q_grp = q_fp32.reshape(actual_s1, n2, g, d)
            qk = torch.einsum('qngd,knd->qngk', q_grp, k_fp32)   # [s1', N2, G, s2']
            qk_relu = torch.relu(qk)
            w_grp = w_fp32.reshape(actual_s1, n2, g).unsqueeze(-1)
            score_sum = (qk_relu * w_grp).sum(dim=2)             # [s1', N2, s2']

            # ---- mask：masked 置 -inf，不参与 topk（对齐 golden） ----
            if sparse_mode == 3:
                i_idx = torch.arange(actual_s1, device=device).unsqueeze(1)
                j_idx = torch.arange(actual_s2, device=device).unsqueeze(0)
                mask_bool = j_idx > (i_idx + (actual_s2 - actual_s1))   # True = 屏蔽
                score_sum = score_sum.masked_fill(mask_bool.unsqueeze(1), float('-inf'))

            # ---- topk：分数降序、同分取下标小者优先（stable sort，对齐 golden 的 lexsort） ----
            topk_count = min(sparse_count, actual_s2)
            if topk_count == 0:
                continue
            sorted_vals, sorted_idx = torch.sort(score_sum, dim=-1, descending=True, stable=True)
            topk_values = sorted_vals[..., :topk_count]
            topk_indices = sorted_idx[..., :topk_count]

            # ---- 有效槽位不足 topk_count 时补 -1 / -inf（对齐 golden） ----
            if sparse_mode == 3:
                rows = torch.arange(actual_s1, device=device)
                valid_len = torch.clamp(rows + (actual_s2 - actual_s1) + 1, min=0, max=actual_s2)
            else:
                valid_len = torch.full((actual_s1,), actual_s2, dtype=torch.long, device=device)
            col_idx = torch.arange(topk_count, device=device)
            beyond = col_idx.unsqueeze(0) >= valid_len.unsqueeze(1)     # [s1', topk_count]
            topk_indices = topk_indices.masked_fill(beyond.unsqueeze(1), -1)
            topk_values = topk_values.masked_fill(beyond.unsqueeze(1), float('-inf'))

            if layout_query == "BSND":
                indices_out[batch_idx, :actual_s1, :, :topk_count] = topk_indices.to(torch.int32)
                values_out[batch_idx, :actual_s1, :, :topk_count] = topk_values
            else:
                indices_out[row0:row0 + actual_s1, :, :topk_count] = topk_indices.to(torch.int32)
                values_out[row0:row0 + actual_s1, :, :topk_count] = topk_values

        if return_value:
            return (indices_out, values_out)
        else:
            return (indices_out, torch.zeros_like(indices_out))
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, weights: torch.Tensor,
                actual_seq_lengths_query=None, actual_seq_lengths_key=None,
                block_table=None, layout_query="BSND", layout_key="BSND",
                sparse_count=2048, sparse_mode=3, pre_tokens=None, next_tokens=None,
                return_value=False):
        """
        Performs lightning indexer computation on NPU.

        Args:
            query (Tensor): Index query tensor, shape [B, S1, N1, D] or [T1, N1, D], dtype bfloat16/float16.
            key (Tensor): Index key tensor, shape [B, S2, N2, D] or PA_BSND/TND, dtype bfloat16/float16.
            weights (Tensor): Weight tensor, shape [B, S1, N1] or [T, N1], dtype bfloat16/float16.
            actual_seq_lengths_query (Tensor, optional): Valid token count per batch for query, dtype int32.
            actual_seq_lengths_key (Tensor, optional): Valid token count per batch for key, dtype int32.
            block_table (Tensor, optional): PageAttention block mapping table, dtype int32.
            layout_query (str): Query data layout, supports BSND/TND, default "BSND".
            layout_key (str): Key data layout, supports PA_BSND/BSND/TND, default "BSND".
            sparse_count (int): Number of blocks to retain in topK, range [1, 2048], default 2048.
            sparse_mode (int): Sparse mode, supports 0/3, default 3.
            pre_tokens (int, optional): Forward token count for sparse computation, default 2^63-1.
            next_tokens (int, optional): Backward token count for sparse computation, default 2^63-1.
            return_value (bool): Whether to output sparse_values, default False.
                Note: For layout_key="PA_BSND", the NPU kernel does not support
                return_value=True. This implementation passes return_value=False
                to the kernel and constructs a zero-filled tensor in Python when
                return_value=False. For other layouts, return_value=True is always
                passed to the kernel to avoid uninitialized memory, then post-processed.

        Returns:
            tuple: (sparse_indices, sparse_values) where sparse_indices is int32 and sparse_values is int32.
        """
        import torch_npu
        if pre_tokens is None:
            pre_tokens = (1 << 63) - 1
        if next_tokens is None:
            next_tokens = (1 << 63) - 1

        # PA_BSND does not support return_value=True; other layouts force True to avoid
        # uninitialized memory (NaN/Inf) when return_value=False.
        if layout_key == "PA_BSND":
            sparse_indices, _ = torch_npu.npu_lightning_indexer(
                query, key, weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table, layout_query=layout_query, layout_key=layout_key,
                sparse_count=sparse_count, sparse_mode=sparse_mode,
                pre_tokens=pre_tokens, next_tokens=next_tokens,
                return_value=False)
            sparse_values = torch.zeros(
                sparse_indices.shape, dtype=torch.int32, device=sparse_indices.device)
        else:
            # Force return_value=True to avoid uninitialized memory bug.
            sparse_indices, sparse_values_raw = torch_npu.npu_lightning_indexer(
                query, key, weights,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                block_table=block_table, layout_query=layout_query, layout_key=layout_key,
                sparse_count=sparse_count, sparse_mode=sparse_mode,
                pre_tokens=pre_tokens, next_tokens=next_tokens,
                return_value=True)
            if not return_value:
                sparse_values = torch.zeros(
                    sparse_indices.shape, dtype=torch.int32, device=sparse_indices.device)
            else:
                sparse_values = sparse_values_raw.to(torch.int32)

        # 输出规范化（方案 A）：top-k 索引按升序输出，使精度比较与排列顺序解耦，
        # torch_npu.npu_lightning_indexer 调用与结果完全不变，仅重排输出顺序。
        perm = torch.argsort(sparse_indices, dim=-1, stable=True)
        sparse_indices = torch.gather(sparse_indices, -1, perm)
        if return_value:
            sparse_values = torch.gather(sparse_values, -1, perm)

        return (sparse_indices, sparse_values)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "58_LightningIndexer.json")
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
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        query_info = inputs[0]
        key_info = inputs[1]
        weights_info = inputs[2]

        dtype = dtype_map[query_info["dtype"]]

        query = random_tensor(query_info["shape"], dtype)
        key = random_tensor(key_info["shape"], dtype)
        weights = random_tensor(weights_info["shape"], dtype)

        actual_seq_lengths_query = None
        actual_seq_lengths_key = None
        block_table = None
        layout_query = "BSND"
        layout_key = "BSND"
        sparse_count = 2048
        sparse_mode = 3
        pre_tokens = None
        next_tokens = None
        return_value = False

        for inp in inputs[3:]:
            name = inp.get("name", "")
            if name == "actual_seq_lengths_query":
                actual_seq_lengths_query = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "actual_seq_lengths_key":
                actual_seq_lengths_key = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "block_table":
                block_shape = inp["shape"]
                total_blocks = block_shape[0] * block_shape[1]
                block_table = torch.arange(total_blocks, dtype=torch.int32).reshape(block_shape)
            elif name == "layout_query":
                layout_query = inp["value"]
            elif name == "layout_key":
                layout_key = inp["value"]
            elif name == "sparse_count":
                sparse_count = inp["value"]
            elif name == "sparse_mode":
                sparse_mode = inp["value"]
            elif name == "pre_tokens":
                pre_tokens = inp["value"]
            elif name == "next_tokens":
                next_tokens = inp["value"]
            elif name == "return_value":
                return_value = inp["value"]

        input_groups.append([query, key, weights,
                             actual_seq_lengths_query, actual_seq_lengths_key,
                             block_table, layout_query, layout_key,
                             sparse_count, sparse_mode, pre_tokens, next_tokens,
                             return_value])
    return input_groups


def get_init_inputs():
    return []