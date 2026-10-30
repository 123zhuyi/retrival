import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs Fusion Attention computation using NPU accelerated npu_fusion_attention.
    Implements fused Transformer Attention Score computation.
    torch_npu.npu_fusion_attention(query, key, value, head_num, input_layout, *, pse=None, padding_mask=None, atten_mask=None, scale=1., keep_prob=1., pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0, prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0, gen_mask_parallel=True, sync=False, softmax_layout="", sink=None) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)
    PyTorch native implementation of forward function
    def _to_bnsd(tensor, layout, head_num):
        if layout == "BNSD":
            return tensor
        if layout == "BSND":
            return tensor.transpose(1, 2).contiguous()
        if layout == "BSH":
            b, s, h = tensor.shape
            d = h // head_num
            return tensor.view(b, s, head_num, d).transpose(1, 2).contiguous()
        if layout == "SBH":
            s, b, h = tensor.shape
            d = h // head_num
            return tensor.view(s, b, head_num, d).permute(1, 2, 0, 3).contiguous()
        raise ValueError(f"Unsupported layout: {layout}")

    def _from_bnsd(tensor, layout, head_num):
        if layout == "BNSD":
            return tensor
        if layout == "BSND":
            return tensor.transpose(1, 2).contiguous()
        if layout == "BSH":
            b, n, s, d = tensor.shape
            return tensor.transpose(1, 2).contiguous().view(b, s, n * d)
        if layout == "SBH":
            b, n, s, d = tensor.shape
            return tensor.permute(2, 0, 1, 3).contiguous().view(s, b, n * d)
        raise ValueError(f"Unsupported layout: {layout}")

    def _generate_default_mask(sq, skv, pre_tockens, next_tockens, device, dtype):
        mask = torch.ones(1, 1, sq, skv, device=device, dtype=dtype)
        for i in range(sq):
            for j in range(skv):
                diff = i - j
                if diff <= next_tockens and diff >= -pre_tockens:
                    mask[0, 0, i, j] = 0
        return mask

    def _generate_leftup_causal_mask(sq, skv, device, dtype):
        mask = torch.ones(1, 1, sq, skv, device=device, dtype=dtype)
        for i in range(sq):
            for j in range(skv):
                if j <= i:
                    mask[0, 0, i, j] = 0
        return mask

    def _generate_rightdown_causal_mask(sq, skv, device, dtype):
        mask = torch.ones(1, 1, sq, skv, device=device, dtype=dtype)
        offset = skv - sq
        for i in range(sq):
            for j in range(skv):
                if j - offset <= i:
                    mask[0, 0, i, j] = 0
        return mask

    def _apply_sparse_mask(scores, sparse_mode, pre_tockens, next_tockens, atten_mask, sq, skv, b, n):
        device = scores.device
        dtype = scores.dtype
        mask_value = -10000.0
        if sparse_mode == 0:
            if atten_mask is not None:
                mask = atten_mask.to(dtype)
            else:
                mask = _generate_default_mask(sq, skv, pre_tockens, next_tockens, device, dtype)
            scores = scores.masked_fill(mask.to(torch.bool), mask_value)
        elif sparse_mode == 1:
            if atten_mask is not None:
                mask = atten_mask.to(dtype)
                scores = scores.masked_fill(mask.to(torch.bool), mask_value)
        elif sparse_mode == 2:
            if atten_mask is not None:
                mask = atten_mask.to(dtype)
            else:
                mask = _generate_leftup_causal_mask(sq, skv, device, dtype)
            scores = scores.masked_fill(mask.to(torch.bool), mask_value)
        elif sparse_mode == 3:
            if atten_mask is not None:
                mask = atten_mask.to(dtype)
            else:
                mask = _generate_rightdown_causal_mask(sq, skv, device, dtype)
            scores = scores.masked_fill(mask.to(torch.bool), mask_value)
        elif sparse_mode == 4:
            if atten_mask is not None:
                mask = atten_mask.to(dtype)
            else:
                mask = _generate_default_mask(sq, skv, pre_tockens, next_tockens, device, dtype)
            scores = scores.masked_fill(mask.to(torch.bool), mask_value)
        return scores

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                head_num: int, input_layout: str, pse=None, padding_mask=None,
                atten_mask=None, scale=1., keep_prob=1., pre_tockens=2147483647,
                next_tockens=2147483647, inner_precise=0, prefix=None,
                actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
        use_fp32 = inner_precise in (0, 1, 2)
        if input_layout in ("BSH", "SBH"):
            q_h = query.shape[-1]
            k_h = key.shape[-1]
            nq = head_num
            nkv = nq * k_h // q_h if q_h != k_h else nq
            dq = q_h // nq
            dkv = k_h // nkv
            q_bnsd = _to_bnsd(query, input_layout, nq)
            k_bnsd = _to_bnsd(key, input_layout, nkv)
            v_bnsd = _to_bnsd(value, input_layout, nkv)
        else:
            q_bnsd = _to_bnsd(query, input_layout, head_num)
            k_bnsd = _to_bnsd(key, input_layout, head_num)
            v_bnsd = _to_bnsd(value, input_layout, head_num)
            nq = head_num
            nkv = head_num
            dq = q_bnsd.shape[-1]
            dkv = k_bnsd.shape[-1]
        b, _, sq, _ = q_bnsd.shape
        _, _, skv, _ = k_bnsd.shape
        _, _, _, dv = v_bnsd.shape
        group_size = nq // nkv
        compute_dtype = torch.float32 if use_fp32 else query.dtype
        q = q_bnsd.to(compute_dtype)
        k = k_bnsd.to(compute_dtype)
        v = v_bnsd.to(compute_dtype)
        if group_size > 1:
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * scale
        if pse is not None:
            pse_c = pse.to(compute_dtype)
            if pse_c.dim() == 4:
                scores = scores + pse_c
            elif pse_c.dim() == 2:
                scores = scores + pse_c.view(1, 1, 1, -1)
            else:
                scores = scores + pse_c
        if sink is not None:
            sink_c = sink.to(compute_dtype)
            scores = scores + sink_c.view(1, -1, 1, 1)
        scores = _apply_sparse_mask(scores, sparse_mode, pre_tockens, next_tockens,
                                    atten_mask, sq, skv, b, nq)
        if inner_precise == 2:
            row_max = scores.max(dim=-1, keepdim=True).values
            row_all_masked = torch.isinf(row_max)
            if row_all_masked.any():
                scores = scores.masked_fill(row_all_masked, 0.0)
        softmax_max = scores.max(dim=-1, keepdim=True).values
        scores_shifted = scores - softmax_max
        scores_exp = torch.exp(scores_shifted)
        softmax_sum = scores_exp.sum(dim=-1, keepdim=True)
        probs = scores_exp / softmax_sum
        if keep_prob < 1.0:
            keep_prob = 1.0
        if keep_prob < 1.0 and keep_prob > 0:
            dropout_mask = torch.rand_like(probs) < keep_prob
            probs = probs * dropout_mask / keep_prob
        attention_out_bnsd = torch.matmul(probs, v)
        if input_layout in ("BSH", "SBH"):
            attention_out = _from_bnsd(attention_out_bnsd, input_layout, nq)
        else:
            attention_out = _from_bnsd(attention_out_bnsd, input_layout, head_num)
        attention_out = attention_out.to(query.dtype)
        softmax_max_out = softmax_max.squeeze(-1).to(torch.float32)
        softmax_sum_out = softmax_sum.squeeze(-1).to(torch.float32)
        reserved = torch.empty(0, device=query.device, dtype=query.dtype)
        seed = 0
        offset = 0
        mask_length = 0
        return (attention_out, softmax_max_out, softmax_sum_out, reserved, seed, offset, mask_length)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                head_num: int, input_layout: str, pse=None, padding_mask=None,
                atten_mask=None, scale=1., keep_prob=1., pre_tockens=2147483647,
                next_tockens=2147483647, inner_precise=0, prefix=None,
                actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
                gen_mask_parallel=True, sync=False, softmax_layout="", sink=None):
        """
        Performs fused attention computation on NPU.

        Args:
            query (Tensor): Query tensor, dtype float16/bfloat16/float32.
            key (Tensor): Key tensor, same dtype as query.
            value (Tensor): Value tensor, same dtype as query.
            head_num (int): Number of attention heads.
            input_layout (str): Data layout, supports BSH/SBH/BSND/BNSD/TND.
            pse (Tensor, optional): Position encoding.
            padding_mask (Tensor, optional): Not supported yet.
            atten_mask (Tensor, optional): Attention mask, dtype bool/uint8.
            scale (float): Scaling factor, default 1.
            keep_prob (float): Dropout keep probability, default 1.
            pre_tockens (int): Sparse computation parameter, default 2147483647.
            next_tockens (int): Sparse computation parameter, default 2147483647.
            inner_precise (int): Precision control, default 0.
            prefix (List[int], optional): Prefix sparse computation parameter.
            actual_seq_qlen (List[int], optional): Cumulative query sequence lengths for varlen.
            actual_seq_kvlen (List[int], optional): Cumulative key/value sequence lengths for varlen.
            sparse_mode (int): Sparse mode, default 0.
            gen_mask_parallel (bool): DSA parallel control, default True.
            sync (bool): DSA sync control, default False.
            softmax_layout (str): Softmax output layout for TND, default "".
            sink (Tensor, optional): Per-head bias, shape [head_num], dtype float32.

        Returns:
            tuple: (attention_out, softmax_max, softmax_sum, reserved, seed, offset, mask_length)
        """
        import torch_npu
        # keep_prob<1 时 npu_fusion_attention 的 dropout 使用 NPU 内部 RNG:
        # seed 跟随全局 generator 但 offset 每次调用自增, 同输入两次输出
        # max_abs_diff=4.13(实测), golden 不是输入的确定函数, 任何实现都无法
        # 逐点复现随机 mask。任务层固定为无 dropout, 保证 golden 确定性。
        if keep_prob < 1.0:
            keep_prob = 1.0
        return torch_npu.npu_fusion_attention(query, key, value, head_num, input_layout,
                                               pse=pse, padding_mask=padding_mask,
                                               atten_mask=atten_mask, scale=scale,
                                               keep_prob=keep_prob, pre_tockens=pre_tockens,
                                               next_tockens=next_tockens, inner_precise=inner_precise,
                                               prefix=prefix, actual_seq_qlen=actual_seq_qlen,
                                               actual_seq_kvlen=actual_seq_kvlen, sparse_mode=sparse_mode,
                                               gen_mask_parallel=gen_mask_parallel, sync=sync,
                                               softmax_layout=softmax_layout, sink=sink)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_FusionAttention.json")
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
        }

        query_info = inputs[0]
        key_info = inputs[1]
        value_info = inputs[2]
        head_num_info = inputs[3]
        input_layout_info = inputs[4]

        dtype = dtype_map[query_info["dtype"]]

        # 每个浮点 tensor 独立随机选择分布
        query = random_tensor(query_info["shape"], dtype)
        key = random_tensor(key_info["shape"], dtype)
        value = random_tensor(value_info["shape"], dtype)
        head_num = head_num_info["value"]
        input_layout = input_layout_info["value"]

        pse = None
        padding_mask = None
        atten_mask = None
        scale = 1.
        keep_prob = 1.
        pre_tockens = 2147483647
        next_tockens = 2147483647
        inner_precise = 0
        prefix = None
        actual_seq_qlen = None
        actual_seq_kvlen = None
        sparse_mode = 0
        gen_mask_parallel = True
        sync = False
        softmax_layout = ""
        sink = None

        for inp in inputs[5:]:
            name = inp.get("name", "")
            if name == "pse":
                pse = random_tensor(inp["shape"], dtype)
            elif name == "padding_mask":
                padding_mask = None
            elif name == "atten_mask":
                atten_mask_dtype_map = {
                    "bool": torch.bool,
                    "uint8": torch.uint8,
                }
                atten_mask_dtype = atten_mask_dtype_map.get(inp.get("dtype", "bool"), torch.bool)
                atten_mask = torch.ones(inp["shape"], dtype=atten_mask_dtype)
            elif name == "scale":
                scale = inp["value"]
            elif name == "keep_prob":
                keep_prob = inp["value"]
            elif name == "pre_tockens":
                pre_tockens = inp["value"]
            elif name == "next_tockens":
                next_tockens = inp["value"]
            elif name == "inner_precise":
                inner_precise = inp["value"]
            elif name == "prefix":
                prefix = inp["value"]
            elif name == "actual_seq_qlen":
                actual_seq_qlen = inp["value"]
            elif name == "actual_seq_kvlen":
                actual_seq_kvlen = inp["value"]
            elif name == "sparse_mode":
                sparse_mode = inp["value"]
            elif name == "gen_mask_parallel":
                gen_mask_parallel = inp["value"]
            elif name == "sync":
                sync = inp["value"]
            elif name == "softmax_layout":
                softmax_layout = inp["value"]
            elif name == "sink":
                sink = random_tensor(inp["shape"], torch.float32)

        input_groups.append([query, key, value, head_num, input_layout,
                             pse, padding_mask, atten_mask, scale, keep_prob,
                             pre_tockens, next_tockens, inner_precise, prefix,
                             actual_seq_qlen, actual_seq_kvlen, sparse_mode,
                             gen_mask_parallel, sync, softmax_layout, sink])
    return input_groups


def get_init_inputs():
    return []